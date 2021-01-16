import kornia.augmentation as K
import torch.nn as nn
from kornia.contrib import ExtractTensorPatches
from torch.nn.modules.utils import _pair

from mmaction.utils import add_prefix
from .. import builder
from ..common import images2video, video2images
from ..registry import TRACKERS
from .vanilla_tracker import VanillaTracker


@TRACKERS.register_module()
class SimSiamBaseTracker(VanillaTracker):
    """3D recognizer model framework."""

    def __init__(self, *args, backbone, img_head=None, **kwargs):
        super().__init__(*args, backbone=backbone, **kwargs)
        if img_head is not None:
            self.img_head = builder.build_head(img_head)
        self.init_extra_weights()
        if self.train_cfg is not None:
            self.intra_video = self.train_cfg.get('intra_video', False)
            self.transpose_temporal = self.train_cfg.get(
                'transpose_temporal', False)
            if self.train_cfg.get('image2patch', False):
                self.patch_size = _pair(self.train_cfg.get('patch_size', 64))
                self.patch_stride = _pair(
                    self.train_cfg.get('patch_stride', 32))
                self.image2patch = nn.Sequential(
                    ExtractTensorPatches(
                        window_size=self.patch_size, stride=self.patch_stride),
                    nn.Flatten(0, 1),
                    K.RandomResizedCrop(
                        size=self.patch_size, scale=(0.7, 0.9)))
            else:
                self.image2patch = None

    @property
    def with_img_head(self):
        """bool: whether the detector has img head"""
        return hasattr(self, 'img_head') and self.img_head is not None

    def init_extra_weights(self):
        if self.with_img_head:
            self.img_head.init_weights()

    def forward_img_head(self, x1, x2, clip_len):
        if isinstance(x1, tuple):
            x1 = x1[-1]
        if isinstance(x2, tuple):
            x2 = x2[-1]
        losses = dict()
        z1, p1 = self.img_head(x1)
        z2, p2 = self.img_head(x2)
        loss_weight = 1. / clip_len if self.intra_video else 1.
        losses.update(
            add_prefix(
                self.img_head.loss(p1, z1, p2, z2, weight=loss_weight),
                prefix='0'))
        if self.intra_video:
            z2_v, p2_v = images2video(z2, clip_len), images2video(p2, clip_len)
            for i in range(1, clip_len):
                losses.update(
                    add_prefix(
                        self.img_head.loss(
                            p1,
                            z1,
                            video2images(p2_v.roll(i, dims=2)),
                            video2images(z2_v.roll(i, dims=2)),
                            weight=loss_weight),
                        prefix=f'{i}'))
        return losses

    def forward_train(self, imgs, grids=None, label=None):
        # [B, N, C, T, H, W]
        if self.transpose_temporal:
            imgs = imgs.transpose(1, 3).contiguous()
        assert imgs.size(1) == 2
        assert imgs.ndim == 6
        clip_len = imgs.size(3)
        imgs1 = video2images(imgs[:,
                                  0].contiguous().reshape(-1, *imgs.shape[2:]))
        imgs2 = video2images(imgs[:,
                                  1].contiguous().reshape(-1, *imgs.shape[2:]))
        if self.image2patch is not None:
            imgs1 = self.image2patch(imgs1)
            imgs2 = self.image2patch(imgs2)
        x1 = self.backbone(imgs1)
        x2 = self.backbone(imgs2)
        losses = dict()
        if self.with_img_head:
            loss_img_head = self.forward_img_head(x1, x2, clip_len)
            losses.update(add_prefix(loss_img_head, prefix='img_head'))

        return losses
