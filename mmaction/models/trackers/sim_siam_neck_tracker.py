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
class SimSiamNeckTracker(VanillaTracker):
    """3D recognizer model framework."""

    def __init__(self,
                 *args,
                 backbone,
                 backbone_head=None,
                 neck_head=None,
                 **kwargs):
        super().__init__(*args, backbone=backbone, **kwargs)
        if backbone_head is not None:
            self.backbone_head = builder.build_head(backbone_head)
        if neck_head is not None:
            self.neck_head = builder.build_head(neck_head)
        self.init_extra_weights()
        if self.train_cfg is not None:
            self.intra_video = self.train_cfg.get('intra_video', False)
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
    def with_backbone_head(self):
        """bool: whether the detector has backbone head"""
        return hasattr(self,
                       'backbone_head') and self.backbone_head is not None

    @property
    def with_neck_head(self):
        """bool: whether the detector has neck head"""
        return hasattr(self, 'neck_head') and self.neck_head is not None

    def init_extra_weights(self):
        if self.with_backbone_head:
            if isinstance(self.backbone_head, nn.Sequential):
                for m in self.backbone_head:
                    m.init_weights()
            else:
                self.backbone_head.init_weights()
        if self.with_neck_head:
            if isinstance(self.neck_head, nn.Sequential):
                for m in self.neck_head:
                    m.init_weights()
            else:
                self.neck_head.init_weights()

    def forward_head(self, head, x1, x2, clip_len):
        losses = dict()
        z1, p1 = head(x1)
        z2, p2 = head(x2)
        loss_weight = 1. / clip_len if self.intra_video else 1.
        losses.update(
            add_prefix(
                head.loss(p1, z1, p2, z2, weight=loss_weight), prefix='0'))
        if self.intra_video:
            z2_v, p2_v = images2video(z2, clip_len), images2video(p2, clip_len)
            for i in range(1, clip_len):
                losses.update(
                    add_prefix(
                        head.loss(
                            p1,
                            z1,
                            video2images(p2_v.roll(i, dims=2)),
                            video2images(z2_v.roll(i, dims=2)),
                            weight=loss_weight),
                        prefix=f'{i}'))
        return losses

    def forward_multi_heads(self, heads, x1, x2, clip_len, prefix):
        losses = dict()
        if not isinstance(heads, nn.Sequential) or len(heads) == 1:
            # use last
            if isinstance(x1, (tuple, list)):
                x1 = x1[-1]
            if isinstance(x2, (tuple, list)):
                x2 = x2[-1]
            loss_head = self.forward_head(heads, x1, x2, clip_len)
            losses.update(add_prefix(loss_head, prefix=prefix))
        else:
            assert len(heads) == len(x1) == len(x2)
            for idx in range(len(heads)):
                loss_head = self.forward_head(heads[idx], x1[idx], x2[idx],
                                              clip_len)
                losses.update(add_prefix(loss_head, prefix=f'{prefix}[{idx}]'))

        return losses

    def forward_train(self, imgs, grids=None, label=None):
        # [B, N, C, T, H, W]
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
        if self.with_backbone_head:
            losses.update(
                self.forward_multi_heads(
                    self.backbone_head,
                    x1,
                    x2,
                    clip_len,
                    prefix='backbone_head'))
        if self.with_neck_head:
            neck_x1 = self.neck(x1)
            neck_x2 = self.neck(x2)
            losses.update(
                self.forward_multi_heads(
                    self.neck_head,
                    neck_x1,
                    neck_x2,
                    clip_len,
                    prefix='neck_head'))

        return losses
