import torch
import torch.nn.functional as F

from mmaction.utils import add_prefix
from .. import builder
from ..common import concat_all_gather, images2video, video2images
from ..registry import TRACKERS
from .vanilla_tracker import VanillaTracker


@TRACKERS.register_module()
class MoCoBaseTracker(VanillaTracker):
    """3D recognizer model framework."""

    def __init__(self,
                 *args,
                 backbone,
                 img_head,
                 queue_dim=128,
                 img_queue_size=65536,
                 **kwargs):
        super().__init__(*args, backbone=backbone, **kwargs)
        self.encoder_k = builder.build_backbone(backbone)
        # create the queue
        self.queue_dim = queue_dim
        if img_head is not None:
            self.img_head_q = builder.build_head(img_head)
            self.img_head_k = builder.build_head(img_head)
            self.img_queue_size = img_queue_size

            # image queue
            self.register_buffer('img_queue',
                                 torch.randn(queue_dim, self.img_queue_size))
            self.img_queue = F.normalize(self.img_queue, dim=0, p=2)
            self.register_buffer('img_queue_ptr',
                                 torch.zeros(1, dtype=torch.long))

        self.init_moco_weights()
        if self.train_cfg is not None:
            self.shuffle_bn = self.train_cfg.get('shuffle_bn', True)
            self.momentum = self.train_cfg.get('momentum', 0.999)

        assert self.with_img_head

    @property
    def with_img_head(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'img_head_q') and self.img_head_q is not None

    @property
    def encoder_q(self):
        return self.backbone

    def extract_encoder_feature(self, encoder, imgs):
        outs = encoder(imgs)
        if isinstance(outs, tuple):
            return outs[-1]
        else:
            return outs

    def init_moco_weights(self):
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        if self.with_img_head:
            self.img_head_q.init_weights()
            for param_q, param_k in zip(self.img_head_q.parameters(),
                                        self.img_head_k.parameters()):
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

    def forward_img_head(self, x_q, x_k, clip_len):
        head_k = self.img_head_k
        head_q = self.img_head_q
        with torch.no_grad():
            full_embed_x_k = head_k(video2images(x_k))
        full_embed_x_q = head_q(video2images(x_q))
        full_embed_x_k = images2video(full_embed_x_k, clip_len)
        full_embed_x_q = images2video(full_embed_x_q, clip_len)
        loss = dict()
        loss_full = self.img_head_q.loss(full_embed_x_q, full_embed_x_k,
                                         self.img_queue.clone().detach())
        loss.update(loss_full)
        # dequeue and enqueue
        self._dequeue_and_enqueue_img(video2images(full_embed_x_k))
        return loss

    def forward_train(self, imgs, label=None):
        """Defines the computation performed at every call when training."""
        # [B, N, C, T, H, W]
        assert imgs.size(1) == 2
        assert imgs.ndim == 6
        clip_len = imgs.size(3)
        # [B, C, T, H, W]
        imgs_k = imgs[:, 0].contiguous().reshape(-1, *imgs.shape[2:])
        imgs_q = imgs[:, 1].contiguous().reshape(-1, *imgs.shape[2:])
        x_q = images2video(self.encoder_q(video2images(imgs_q)), clip_len)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            if self.shuffle_bn:
                # shuffle for making use of BN
                imgs_k_shuffled, idx_unshuffle = self._batch_shuffle_ddp(
                    imgs_k)

                # [N, C, T, H, W]
                x_k = images2video(
                    self.encoder_k(video2images(imgs_k_shuffled)), clip_len)
                # undo shuffle
                x_k = self._batch_unshuffle_ddp(x_k, idx_unshuffle)
            else:
                # [N, C, T, H, W]
                x_k = images2video(
                    self.encoder_k(video2images(imgs_k)), clip_len)
            assert x_k.size() == x_q.size()
        loss = dict()
        if self.with_img_head:
            loss.update(
                add_prefix(
                    self.forward_img_head(x_q, x_k, clip_len),
                    prefix='img_head'))
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
