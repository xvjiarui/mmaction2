import torch.nn.functional as F

from mmaction.utils import add_prefix
from .. import builder
from ..common import (grid_mask, images2video, masked_attention_efficient,
                      video2images)
from ..registry import TRACKERS
from .vanilla_tracker import VanillaTracker


@TRACKERS.register_module()
class SimSiamPixTracker(VanillaTracker):
    """3D recognizer model framework."""

    def __init__(self,
                 *args,
                 backbone,
                 pix_head=None,
                 img_head=None,
                 **kwargs):
        super().__init__(*args, backbone=backbone, **kwargs)
        if pix_head is not None:
            self.pix_head = builder.build_head(pix_head)
        if img_head is not None:
            self.img_head = builder.build_head(img_head)
        self.init_extra_weights()
        if self.train_cfg is not None:
            self.intra_video = self.train_cfg.get('intra_video', False)
            self.patch_grid_radius = self.train_cfg.get(
                'patch_grid_radius', None)
            self.patch_att_mode = self.train_cfg.get('patch_att_mode',
                                                     'cosine')
            self.cls_on_pix = self.train_cfg.get('cls_on_pix', False)

    @property
    def with_pix_head(self):
        """bool: whether the detector has patch head"""
        return hasattr(self, 'pix_head') and self.pix_head is not None

    @property
    def with_img_head(self):
        """bool: whether the detector has img head"""
        return hasattr(self, 'img_head') and self.img_head is not None

    def init_extra_weights(self):
        if self.with_img_head:
            self.img_head.init_weights()
        if self.with_pix_head:
            self.pix_head.init_weights()

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

    def forward_cls_head(self, x1, x2, clip_len):
        losses = dict()
        z1, p1 = self.cls_head(x1)
        z2, p2 = self.cls_head(x2)
        loss_weight = 1. / clip_len if self.intra_video else 1.
        losses.update(
            add_prefix(
                self.cls_head.loss(p1, z1, p2, z2, weight=loss_weight),
                prefix='0'))
        if self.intra_video:
            z2_v, p2_v = images2video(z2, clip_len), images2video(p2, clip_len)
            for i in range(1, clip_len):
                losses.update(
                    add_prefix(
                        self.cls_head.loss(
                            p1,
                            z1,
                            video2images(p2_v.roll(i, dims=2)),
                            video2images(z2_v.roll(i, dims=2)),
                            weight=loss_weight),
                        prefix=f'{i}'))
        return losses

    def forward_pix_head(self, x1, x2, clip_len, grids1=None, grids2=None):
        x1 = masked_attention_efficient(
            x1, x2, x2, mask=None, mode=self.patch_att_mode)
        x2 = masked_attention_efficient(
            x2, x1, x1, mask=None, mode=self.patch_att_mode)

        loss_weight = 1. / clip_len if self.intra_video else 1.
        losses = dict()
        z1, p1 = self.pix_head(x1)
        z2, p2 = self.pix_head(x2)
        if self.patch_grid_radius is not None:
            x_grid1 = F.interpolate(
                grids1, x1.shape[2:], mode='bilinear', align_corners=False)
            x_grid2 = F.interpolate(
                grids2, x2.shape[2:], mode='bilinear', align_corners=False)
            mask12 = grid_mask(x_grid1, x_grid2, self.patch_grid_radius)
            mask21 = mask12.transpose(1, 2)
        else:
            mask12 = None
            mask21 = None
        losses.update(
            add_prefix(
                self.pix_head.loss(
                    p1, z1, p2, z2, mask12, mask21, weight=loss_weight),
                prefix='0.0'))
        if self.intra_video:
            z2_v, p2_v = images2video(z2, clip_len), images2video(p2, clip_len)
            if self.patch_grid_radius:
                mask12_v, mask21_v = images2video(mask12,
                                                  clip_len), images2video(
                                                      mask21, clip_len)
            for i in range(1, clip_len):
                losses.update(
                    add_prefix(
                        self.pix_head.loss(
                            p1,
                            z1,
                            video2images(p2_v.roll(i, dims=2)),
                            video2images(z2_v.roll(i, dims=2)),
                            video2images(mask12_v.roll(i, dims=2))
                            if self.patch_grid_radius else None,
                            video2images(mask21_v.roll(i, dims=2))
                            if self.patch_grid_radius else None,
                            weight=loss_weight),
                        prefix=f'0.{i}'))

        if self.with_cls_head and self.cls_on_pix:
            loss_cls_head = self.forward_cls_head(x1, x2, clip_len)
            losses.update(add_prefix(loss_cls_head, prefix='cls_head'))

        return losses

    def forward_train(self, imgs, grids=None, label=None):
        # [B, N, C, T, H, W]
        assert imgs.size(1) == 2
        clip_len = imgs.size(3)
        imgs1 = video2images(imgs[:,
                                  0].contiguous().reshape(-1, *imgs.shape[2:]))
        imgs2 = video2images(imgs[:,
                                  1].contiguous().reshape(-1, *imgs.shape[2:]))
        if grids is not None:
            grids1 = video2images(grids[:, 0].contiguous().reshape(
                -1, *grids.shape[2:]))
            grids2 = video2images(grids[:, 1].contiguous().reshape(
                -1, *grids.shape[2:]))
        else:
            grids1 = None
            grids2 = None
        x1 = self.backbone(imgs1)
        x2 = self.backbone(imgs2)
        losses = dict()
        if self.with_img_head:
            loss_img_head = self.forward_img_head(x1, x2, clip_len)
            losses.update(add_prefix(loss_img_head, prefix='img_head'))
        if self.with_pix_head or self.with_cls_head:
            neck_x1 = self.neck(x1)
            neck_x2 = self.neck(x2)
        if self.with_pix_head:
            loss_patch_head = self.forward_pix_head(neck_x1, neck_x2, clip_len,
                                                    grids1, grids2)
            losses.update(add_prefix(loss_patch_head, prefix='pix_head'))
        if self.with_cls_head and not self.cls_on_pix:
            loss_cls_head = self.forward_cls_head(neck_x1, neck_x2, clip_len)
            losses.update(add_prefix(loss_cls_head, prefix='cls_head'))

        return losses
