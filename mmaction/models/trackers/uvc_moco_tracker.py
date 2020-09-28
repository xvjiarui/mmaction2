import kornia.augmentation as K
import numpy.random as npr
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmaction.utils import add_suffix
from .. import builder
from ..common import (concat_all_gather, crop_and_resize, get_crop_grid,
                      get_random_crop_bbox, images2video, video2images)
from ..registry import WALKERS
from .vanilla_tracker import VanillaTracker


@WALKERS.register_module()
class UVCMoCoTracker(VanillaTracker):
    """3D recognizer model framework."""

    def __init__(self, *args, backbone, **kwargs):
        super().__init__(*args, backbone=backbone, **kwargs)
        self.stride = self.backbone.output_stride
        self.encoder_k = builder.build_backbone(backbone)
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        if self.train_cfg is not None:
            self.patch_img_size = _pair(self.train_cfg.patch_size)
            self.patch_x_size = _pair(self.train_cfg.patch_size // self.stride)
            if self.train_cfg.get('strong_aug', False):
                same_on_batch = self.train_cfg.get('same_on_batch', False)
                self.aug = nn.Sequential(
                    K.RandomRotation(degrees=10, same_on_batch=same_on_batch),
                    K.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.1,
                        same_on_batch=same_on_batch))
            else:
                self.aug = nn.Identity()
            self.skip_cycle = self.train_cfg.get('skip_cycle', False)
            self.cur_as_tar = self.train_cfg.get('cur_as_tar', False)
            self.shuffle_bn = self.train_cfg.get('shuffle_bn', False)
            self.momentum = self.train_cfg.get('momentum', 0.999)

    @property
    def encoder_q(self):
        return self.backbone

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

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

    def crop_x_from_img(self, img, x, bboxes, crop_first):
        assert isinstance(crop_first, (bool, float))
        if isinstance(crop_first, float):
            crop_first = npr.rand() < crop_first
        if crop_first:
            crop_x = self.extract_feat(
                crop_and_resize(img, bboxes * self.stride,
                                self.patch_img_size))
        else:
            crop_x = crop_and_resize(x, bboxes, self.patch_x_size)

        return crop_x

    def track(self, tar_frame, tar_q_x, tar_k_x, ref_crop_x, tar_bboxes=None):
        if tar_bboxes is None:
            tar_bboxes = self.cls_head.get_tar_bboxes(ref_crop_x, tar_k_x)
        tar_crop_x = self.crop_x_from_img(tar_frame, tar_q_x, tar_bboxes,
                                          self.train_cfg.img_as_tar)

        return tar_bboxes, tar_crop_x

    def forward_train(self, imgs, labels=None):
        """Defines the computation performed at every call when training."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        batches, clip_len = imgs.size(0), imgs.size(2)
        q_x = images2video(
            self.encoder_q(self.aug(video2images(imgs))), clip_len)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            if self.shuffle_bn:
                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(imgs)

                k_x = images2video(
                    self.encoder_k(self.aug(video2images(im_k))), clip_len)

                # undo shuffle
                k_x = self._batch_unshuffle_ddp(k_x, idx_unshuffle)
            else:
                k_x = images2video(
                    self.encoder_k(self.aug(video2images(imgs))), clip_len)
        loss = dict()
        for step in range(2, clip_len + 1):
            # step_weight = 1. if step == 2 or self.iteration > 1000 else 0
            step_weight = 1.
            skip_weight = 1.
            ref_frame = imgs[:, :, 0]
            ref_x = q_x[:, :, 0]
            # all bboxes are in feature space
            ref_bboxes, is_center_crop = get_random_crop_bbox(
                batches,
                self.patch_x_size,
                ref_x.shape[2:],
                device=q_x.device,
                center_ratio=self.train_cfg.center_ratio)
            ref_crop_x = self.crop_x_from_img(ref_frame, ref_x, ref_bboxes,
                                              self.train_cfg.img_as_ref)
            ref_crop_grid = get_crop_grid(ref_frame, ref_bboxes * self.stride,
                                          self.patch_img_size)
            forward_hist = [(ref_bboxes, ref_crop_x)]
            for tar_idx in range(1, step):
                last_bboxes, last_crop_x = forward_hist[-1]
                tar_frame = imgs[:, :, tar_idx]
                tar_q_x = q_x[:, :, tar_idx]
                tar_k_x = k_x[:, :, tar_idx]
                tar_bboxes, tar_crop_x = self.track(
                    tar_frame,
                    tar_q_x,
                    tar_k_x,
                    last_crop_x,
                    tar_bboxes=ref_bboxes if is_center_crop else None)
                forward_hist.append((tar_bboxes, tar_crop_x))
            assert len(forward_hist) == step

            backward_hist = [forward_hist[-1]]
            for last_idx in reversed(range(1, step)):
                tar_idx = last_idx - 1
                last_bboxes, last_crop_x = backward_hist[-1]
                tar_frame = imgs[:, :, tar_idx]
                tar_q_x = q_x[:, :, tar_idx]
                tar_k_x = k_x[:, :, tar_idx]
                tar_bboxes, tar_crop_x = self.track(
                    tar_frame,
                    tar_q_x,
                    tar_k_x,
                    last_crop_x,
                    tar_bboxes=ref_bboxes if is_center_crop else None)
                backward_hist.append((tar_bboxes, tar_crop_x))
            assert len(backward_hist) == step

            loss_step = dict()
            ref_pred_bboxes = backward_hist[-1][0]
            ref_pred_crop_grid = get_crop_grid(ref_frame,
                                               ref_pred_bboxes * self.stride,
                                               self.patch_img_size)
            loss_step['dist_bbox'] = self.cls_head.loss_bbox(
                ref_pred_bboxes / self.patch_x_size[0],
                ref_bboxes / self.patch_x_size[0])
            loss_step['loss_bbox'] = self.cls_head.loss_bbox(
                ref_crop_grid, ref_pred_crop_grid) * step_weight

            for tar_idx in range(1, step):
                last_crop_x = forward_hist[tar_idx - 1][1]
                tar_crop_x = forward_hist[tar_idx][1]
                loss_step.update(
                    add_suffix(
                        self.cls_head.loss(
                            last_crop_x, tar_crop_x, weight=step_weight),
                        suffix=f'forward.t{tar_idx}'))
            for last_idx in reversed(range(1, step)):
                tar_crop_x = backward_hist[last_idx - 1][1]
                last_crop_x = backward_hist[last_idx][1]
                loss_step.update(
                    add_suffix(
                        self.cls_head.loss(
                            tar_crop_x, last_crop_x, weight=step_weight),
                        suffix=f'backward.t{last_idx}'))
            loss.update(add_suffix(loss_step, f'step{step}'))

            if self.cur_as_tar:
                total_hist = forward_hist + backward_hist
                idx_hist = list(range(step)) + list(reversed(range(step)))
                for cur_idx in range(2 * step):
                    loss_cur = dict()
                    cur_bboxes, cur_crop_x = total_hist[cur_idx]
                    cur_frame = imgs[:, :, idx_hist[cur_idx]]
                    cur_x = q_x[:, :, idx_hist[cur_idx]]
                    tar_bboxes, tar_crop_x = self.track(
                        cur_frame, cur_x, cur_crop_x)
                    loss_cur.update(
                        add_suffix(
                            self.cls_head.loss(cur_crop_x, tar_crop_x),
                            suffix='forward'))
                    cur_pred_bboxes, cur_pred_crop_x = self.track(
                        cur_frame, cur_x, tar_crop_x)
                    loss_cur.update(
                        add_suffix(
                            self.cls_head.loss(cur_pred_crop_x, tar_crop_x),
                            suffix='backward'))
                    cur_pred_crop_grid = get_crop_grid(
                        cur_frame, cur_pred_bboxes * self.stride,
                        self.patch_img_size)
                    cur_crop_grid = get_crop_grid(cur_frame,
                                                  cur_bboxes * self.stride,
                                                  self.patch_img_size)
                    loss_cur['loss_bbox'] = self.cls_head.loss_bbox(
                        cur_pred_crop_grid, cur_crop_grid) * step_weight
                    loss.update(add_suffix(loss_cur, f'cur{cur_idx}'))

            if self.skip_cycle and step > 2:
                loss_skip = dict()
                tar_frame = imgs[:, :, step - 1]
                tar_q_x = q_x[:, :, step - 1]
                _, tar_crop_x = self.track(
                    tar_frame,
                    tar_q_x,
                    ref_crop_x,
                    tar_bboxes=ref_bboxes if is_center_crop else None)
                ref_pred_bboxes, ref_pred_crop_x = self.track(
                    ref_frame,
                    ref_x,
                    tar_crop_x,
                    tar_bboxes=ref_bboxes if is_center_crop else None)
                ref_pred_crop_grid = get_crop_grid(
                    ref_frame, ref_pred_bboxes * self.stride,
                    self.patch_img_size)
                loss_skip['dist_bbox'] = self.cls_head.loss_bbox(
                    ref_pred_bboxes / self.patch_x_size[0],
                    ref_bboxes / self.patch_x_size[0])
                loss_skip['loss_bbox'] = self.cls_head.loss_bbox(
                    ref_crop_grid, ref_pred_crop_grid) * skip_weight
                loss_skip.update(
                    add_suffix(
                        self.cls_head.loss(
                            ref_crop_x, tar_crop_x, weight=skip_weight),
                        suffix='forward'))
                loss_skip.update(
                    add_suffix(
                        self.cls_head.loss(
                            tar_crop_x, ref_pred_crop_x, weight=skip_weight),
                        suffix='backward'))
                loss.update(add_suffix(loss_skip, f'skip{step}'))

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
