import kornia.augmentation as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmaction.utils import add_suffix
from .. import builder
from ..common import (StrideContext, bbox_overlaps, concat_all_gather,
                      crop_and_resize, get_crop_grid, get_random_crop_bbox,
                      get_top_diff_crop_bbox, images2video, video2images)
from ..registry import WALKERS
from .vanilla_tracker import VanillaTracker


@WALKERS.register_module()
class UVCMoCoTracker(VanillaTracker):
    """3D recognizer model framework."""

    def __init__(self,
                 *args,
                 backbone,
                 img_head,
                 patch_head,
                 queue_dim=128,
                 img_queue_size=65536,
                 patch_queue_size=65536,
                 **kwargs):
        super().__init__(*args, backbone=backbone, **kwargs)
        self.stride = self.backbone.output_stride
        self.encoder_k = builder.build_backbone(backbone)
        self.with_img_head = img_head is not None
        self.patch_head_q = builder.build_head(patch_head)
        self.patch_head_k = builder.build_head(patch_head)
        if self.with_img_head:
            self.img_head_q = builder.build_head(img_head)
            self.img_head_k = builder.build_head(img_head)
        # create the queue
        self.queue_dim = queue_dim
        if self.with_img_head:
            self.img_queue_size = img_queue_size
        self.patch_queue_size = patch_queue_size

        # image queue
        if self.with_img_head:
            self.register_buffer('img_queue',
                                 torch.randn(queue_dim, img_queue_size))
            self.img_queue = F.normalize(self.img_queue, dim=0, p=2)
            self.register_buffer('img_queue_ptr',
                                 torch.zeros(1, dtype=torch.long))

        # patch queue
        self.register_buffer('patch_queue',
                             torch.randn(queue_dim, patch_queue_size))
        self.patch_queue = F.normalize(self.patch_queue, dim=0, p=2)
        self.register_buffer('patch_queue_ptr',
                             torch.zeros(1, dtype=torch.long))

        self.init_moco_weights()
        if self.train_cfg is not None:
            self.patch_img_size = _pair(self.train_cfg.patch_size)
            self.patch_x_size = _pair(self.train_cfg.patch_size // self.stride)
            if self.train_cfg.get('geo_aug', False):
                self.geo_aug = nn.Sequential(
                    K.RandomRotation(degrees=10),
                    K.RandomHorizontalFlip(p=0.5))
            else:
                self.geo_aug = nn.Identity()
            self.skip_cycle = self.train_cfg.get('skip_cycle', False)
            self.border = self.train_cfg.get('border', 0)
            self.grid_size = self.train_cfg.get('grid_size', 9)
            self.diff_crop = self.train_cfg.get('diff_crop', False)
            self.img_as_grid = self.train_cfg.get('img_as_grid', True)
            self.shuffle_bn = self.train_cfg.get('shuffle_bn', False)
            self.momentum = self.train_cfg.get('momentum', 0.999)
            self.embed_strides = self.train_cfg.get('embed_strides', None)
            self.img_as_ref = self.train_cfg.get('img_as_ref')
            self.img_as_tar = self.train_cfg.get('img_as_tar')
            self.img_as_embed = self.train_cfg.get('img_as_embed')

    @property
    def encoder_q(self):
        return self.backbone

    def init_moco_weights(self):
        if self.with_img_head:
            self.img_head_q.init_weights()
        self.patch_head_q.init_weights()
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        if self.with_img_head:
            for param_q, param_k in zip(self.img_head_q.parameters(),
                                        self.img_head_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.patch_head_q.parameters(),
                                    self.patch_head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)
        if self.with_img_head:
            for param_q, param_k in zip(self.img_head_q.parameters(),
                                        self.img_head_k.parameters()):
                param_k.data = param_k.data * self.momentum + \
                               param_q.data * (1. - self.momentum)
        for param_q, param_k in zip(self.patch_head_q.parameters(),
                                    self.patch_head_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue_img(self, img_keys):
        # gather keys before updating queue
        img_keys = concat_all_gather(img_keys)

        batch_size = img_keys.size(0)
        assert self.img_queue_size % batch_size == 0  # for simplicity

        img_ptr = int(self.img_queue_ptr)

        # replace the img keys at ptr (dequeue and enqueue)
        self.img_queue[:, img_ptr:img_ptr + batch_size] = img_keys.T
        img_ptr = (img_ptr + batch_size) % self.img_queue_size  # move pointer

        self.img_queue_ptr[0] = img_ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_patch(self, patch_keys):
        # gather keys before updating queue
        patch_keys = concat_all_gather(patch_keys)

        batch_size = patch_keys.size(0)
        assert self.patch_queue_size % batch_size == 0  # for simplicity

        patch_ptr = int(self.patch_queue_ptr)

        # replace the patch keys at ptr (dequeue and enqueue)
        self.patch_queue[:, patch_ptr:patch_ptr + batch_size] = patch_keys.T
        patch_ptr = (patch_ptr + batch_size) % self.patch_queue_size  # move
        # pointer

        self.patch_queue_ptr[0] = patch_ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BatchNorm.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def crop_x_from_img(self,
                        img,
                        x,
                        bboxes,
                        encoder,
                        crop_first,
                        trans=None,
                        shuffle_bn=False):
        if crop_first:
            crop_img = crop_and_resize(img, bboxes * self.stride,
                                       self.patch_img_size)
            if trans is not None:
                crop_img = trans(crop_img)
            if shuffle_bn:
                # shuffle for making use of BN
                crop_img_shuffled, idx_unshuffle = self._batch_shuffle_ddp(
                    crop_img)

                # [N, C, H, W]
                crop_x = encoder(crop_img_shuffled)

                # undo shuffle
                crop_x = self._batch_unshuffle_ddp(crop_x, idx_unshuffle)
            else:
                crop_x = encoder(crop_img)
        else:
            crop_x = crop_and_resize(x, bboxes * self.stride,
                                     self.patch_img_size)
            if trans is not None:
                crop_x = trans(crop_x)

        return crop_x

    def get_grid(self, frame, x, bboxes):
        if self.img_as_grid:
            crop_grid = get_crop_grid(frame, bboxes * self.stride,
                                      self.patch_img_size)
        else:
            crop_grid = get_crop_grid(x, bboxes, self.patch_x_size)

        return crop_grid

    def track(self, tar_frame, tar_x, ref_crop_x, *, ref_bboxes=None):
        tar_bboxes = self.cls_head.get_tar_bboxes(ref_crop_x, tar_x,
                                                  ref_bboxes)
        tar_crop_x = self.crop_x_from_img(tar_frame, tar_x, tar_bboxes,
                                          self.encoder_q, self.img_as_tar)

        return tar_bboxes, tar_crop_x

    def get_ref_crop_bbox(self, batches, imgs, idx=0):
        if self.diff_crop:
            ref_bboxes = get_top_diff_crop_bbox(
                imgs[:, :, idx].contiguous(),
                imgs[:, :, (idx + 1) % imgs.size(2)].contiguous(),
                self.patch_img_size,
                self.grid_size,
                device=imgs.device)
        else:
            ref_bboxes, _ = get_random_crop_bbox(
                batches,
                self.patch_img_size,
                imgs.shape[2:],
                device=imgs.device,
                center_ratio=0,
                border=self.border)
        return ref_bboxes / self.stride

    def forward_train(self, imgs, labels=None):
        """Defines the computation performed at every call when training."""
        # [N, C, T, H, W]
        assert imgs.size(1) == 2
        imgs_k = imgs[:, 0].contiguous().reshape(-1, *imgs.shape[2:])
        imgs_q = imgs[:, 1].contiguous().reshape(-1, *imgs.shape[2:])
        # imgs = imgs.reshape((-1,) + imgs.shape[2:])
        batches, clip_len = imgs_q.size(0), imgs_q.size(2)
        x_q = self.encoder_q(video2images(imgs_q))
        if self.with_img_head:
            q_embed_x = self.img_head_q(x_q)
        x_q = images2video(x_q, clip_len)
        if self.with_img_head:
            q_embed_x = images2video(q_embed_x, clip_len)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            if self.shuffle_bn and self.with_img_head:
                # shuffle for making use of BN
                imgs_k_shuffled, idx_unshuffle = self._batch_shuffle_ddp(
                    imgs_k)

                # [N, C, T, H, W]
                x_k = self.encoder_k(video2images(imgs_k_shuffled))
                if self.with_img_head:
                    embed_x_k = self.img_head_k(x_k)
                x_k = images2video(x_k, clip_len)
                if self.with_img_head:
                    embed_x_k = images2video(embed_x_k, clip_len)

                # undo shuffle
                x_k = self._batch_unshuffle_ddp(x_k, idx_unshuffle)
                if self.with_img_head:
                    embed_x_k = self._batch_unshuffle_ddp(
                        embed_x_k, idx_unshuffle)
            else:
                # [N, C, T, H, W]
                x_k = self.encoder_k(video2images(imgs_k))
                if self.with_img_head:
                    embed_x_k = self.img_head_k(x_k)
                x_k = images2video(x_k, clip_len)
                if self.with_img_head:
                    embed_x_k = images2video(embed_x_k, clip_len)
            assert x_k.size() == x_q.size()
        loss = dict()
        patch_embed_x_k = []
        patch_embed_x_q = []
        for step in range(2, clip_len + 1):
            # step_weight = 1. if step == 2 or self.iteration > 1000 else 0
            ref_frame = imgs_q[:, :, 0].contiguous()
            ref_x = x_q[:, :, 0].contiguous()
            # TODO: all bboxes are in feature space
            ref_bboxes = self.get_ref_crop_bbox(batches, imgs_q)
            ref_crop_x = self.crop_x_from_img(
                ref_frame,
                ref_x,
                ref_bboxes,
                self.encoder_q,
                crop_first=self.img_as_ref)
            ref_crop_grid = self.get_grid(ref_frame, ref_x, ref_bboxes)
            forward_hist = [(ref_bboxes, ref_crop_x)]
            for tar_idx in range(1, step):
                last_bboxes, last_crop_x = forward_hist[-1]
                tar_frame = imgs_q[:, :, tar_idx].contiguous()
                tar_x = x_q[:, :, tar_idx].contiguous()
                tar_bboxes, tar_crop_x = self.track(
                    tar_frame, tar_x, last_crop_x, ref_bboxes=last_bboxes)
                forward_hist.append((tar_bboxes, tar_crop_x))
            assert len(forward_hist) == step

            backward_hist = [forward_hist[-1]]
            for tar_idx in reversed(range(step - 1)):
                last_bboxes, last_crop_x = backward_hist[-1]
                tar_frame = imgs_q[:, :, tar_idx].contiguous()
                tar_x = x_q[:, :, tar_idx].contiguous()
                tar_bboxes, tar_crop_x = self.track(
                    tar_frame, tar_x, last_crop_x, ref_bboxes=last_bboxes)
                backward_hist.append((tar_bboxes, tar_crop_x))
            assert len(backward_hist) == step

            loss_step = dict()
            ref_pred_bboxes = backward_hist[-1][0]
            ref_pred_crop_grid = self.get_grid(ref_frame, ref_x,
                                               ref_pred_bboxes)
            loss_step['iou_bbox'] = bbox_overlaps(
                ref_pred_bboxes, ref_bboxes, is_aligned=True)
            loss_step['loss_bbox'] = self.cls_head.loss_bbox(
                ref_crop_grid, ref_pred_crop_grid)

            for tar_idx in range(1, step):
                last_crop_x = forward_hist[tar_idx - 1][1]
                tar_crop_x = forward_hist[tar_idx][1]
                loss_step.update(
                    add_suffix(
                        self.cls_head.loss(last_crop_x, tar_crop_x),
                        suffix=f'forward.t{tar_idx}'))
            for last_idx in reversed(range(1, step)):
                tar_crop_x = backward_hist[last_idx - 1][1]
                last_crop_x = backward_hist[last_idx][1]
                loss_step.update(
                    add_suffix(
                        self.cls_head.loss(tar_crop_x, last_crop_x),
                        suffix=f'backward.t{last_idx}'))

            for idx in range(step):
                last_bboxes, last_crop_x_q = forward_hist[idx]
                with torch.no_grad():
                    with StrideContext(self.encoder_k, self.embed_strides):
                        last_crop_x_k = self.crop_x_from_img(
                            imgs_k[:, :, idx].contiguous(),
                            x_k[:, :, idx].contiguous(),
                            last_bboxes,
                            self.encoder_k,
                            crop_first=self.img_as_embed,
                            trans=self.geo_aug,
                            shuffle_bn=self.shuffle_bn)
                    patch_embed_x_k.append(self.patch_head_k(last_crop_x_k))
                if self.embed_strides is not None or self.img_as_embed:
                    with StrideContext(self.encoder_q, self.embed_strides):
                        last_crop_x_q = self.crop_x_from_img(
                            imgs_q[:, :, idx].contiguous(),
                            x_q[:, :, idx].contiguous(),
                            last_bboxes,
                            self.encoder_q,
                            crop_first=self.img_as_embed,
                            trans=self.geo_aug)
                patch_embed_x_q.append(self.patch_head_q(last_crop_x_q))
            for idx in reversed(range(step - 1)):
                last_bboxes, last_crop_x_q = backward_hist[idx]
                with torch.no_grad():
                    with StrideContext(self.encoder_k, self.embed_strides):
                        last_crop_x_k = self.crop_x_from_img(
                            imgs_k[:, :, idx].contiguous(),
                            x_k[:, :, idx].contiguous(),
                            last_bboxes,
                            self.encoder_k,
                            crop_first=self.img_as_embed,
                            trans=self.geo_aug,
                            shuffle_bn=self.shuffle_bn)
                    patch_embed_x_k.append(self.patch_head_k(last_crop_x_k))
                if self.embed_strides is not None or self.img_as_embed:
                    with StrideContext(self.encoder_q, self.embed_strides):
                        last_crop_x_q = self.crop_x_from_img(
                            imgs_q[:, :, idx].contiguous(),
                            x_q[:, :, idx].contiguous(),
                            last_bboxes,
                            self.encoder_q,
                            crop_first=self.img_as_embed,
                            trans=self.geo_aug)
                patch_embed_x_q.append(self.patch_head_q(last_crop_x_q))

            loss.update(add_suffix(loss_step, f'step{step}'))

            if self.skip_cycle and step > 2:
                loss_skip = dict()
                tar_frame = imgs_q[:, :, step - 1].contiguous()
                tar_x = x_q[:, :, step - 1].contiguous()
                _, tar_crop_x = self.track(tar_frame, tar_x, ref_crop_x)
                ref_pred_bboxes, ref_pred_crop_x = self.track(
                    ref_frame, ref_x, tar_crop_x)
                ref_pred_crop_grid = self.get_grid(ref_frame, ref_x,
                                                   ref_pred_bboxes)
                loss_step['iou_bbox'] = bbox_overlaps(
                    ref_pred_bboxes, ref_bboxes, is_aligned=True)
                loss_skip['loss_bbox'] = self.cls_head.loss_bbox(
                    ref_crop_grid, ref_pred_crop_grid)
                loss_skip.update(
                    add_suffix(
                        self.cls_head.loss(ref_crop_x, tar_crop_x),
                        suffix='forward'))
                loss_skip.update(
                    add_suffix(
                        self.cls_head.loss(tar_crop_x, ref_pred_crop_x),
                        suffix='backward'))
                loss.update(add_suffix(loss_skip, f'skip{step}'))
        patch_embed_x_k = torch.stack(patch_embed_x_k, dim=2)
        patch_embed_x_q = torch.stack(patch_embed_x_q, dim=2)
        if self.with_img_head:
            loss_img = self.img_head_q.loss(q_embed_x, embed_x_k,
                                            self.img_queue)
            loss.update(add_suffix(loss_img, 'img'))
        loss_patch = self.patch_head_q.loss(patch_embed_x_q, patch_embed_x_k,
                                            self.patch_queue)
        loss.update(add_suffix(loss_patch, 'patch'))
        # dequeue and enqueue
        if self.with_img_head:
            self._dequeue_and_enqueue_img(video2images(embed_x_k))
        self._dequeue_and_enqueue_patch(video2images(patch_embed_x_k))
        return loss

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        outs = (self.cls_head(x), )
        return outs
