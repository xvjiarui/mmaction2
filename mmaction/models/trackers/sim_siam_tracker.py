from mmcv.ops import RoIAlign
from torch.nn.modules.utils import _pair

from mmaction.utils import add_prefix, tuple_divide
from .. import builder
from ..common import (bbox2roi, crop_and_resize, get_random_crop_bbox,
                      images2video, masked_attention_efficient,
                      resize_spatial_mask, spatial_neighbor, video2images)
from ..registry import TRACKERS
from .vanilla_tracker import VanillaTracker


@TRACKERS.register_module()
class SimSiamTracker(VanillaTracker):
    """3D recognizer model framework."""

    def __init__(self,
                 *args,
                 backbone,
                 patch_head=None,
                 img_head=None,
                 **kwargs):
        super().__init__(*args, backbone=backbone, **kwargs)
        if patch_head is not None:
            self.patch_head = builder.build_head(patch_head)
        if img_head is not None:
            self.img_head = builder.build_head(img_head)
        self.init_extra_weights()
        if self.train_cfg is not None:
            self.intra_video = self.train_cfg.get('intra_video', False)
            self.patch_size = _pair(self.train_cfg.get('patch_size', 96))
            self.patch_from_img = self.train_cfg.get('patch_from_img', True)
            self.patch_mask_radius = self.train_cfg.get(
                'patch_mask_radius', None)
            self.patch_att_mode = self.train_cfg.get('patch_att_mode',
                                                     'cosine')
            self.patch_tracking = self.train_cfg.get('patch_tracking', True)
            self.patch_cycle_tracking = self.train_cfg.get(
                'patch_cycle_tracking', False)
            self.cls_on_patch = self.train_cfg.get('cls_on_patch', True)
            self.patch_num = self.train_cfg.get('patch_num', 1)
            if self.patch_from_img:
                output_size = self.patch_size
                spatial_scale = 1.
            else:
                output_size = tuple_divide(self.patch_size, 8)
                spatial_scale = 1. / 8.
            if self.patch_num > 1:
                self.roi_align = RoIAlign(
                    output_size=output_size, spatial_scale=spatial_scale)

    @property
    def with_patch_head(self):
        """bool: whether the detector has patch head"""
        return hasattr(self, 'patch_head') and self.patch_head is not None

    @property
    def with_img_head(self):
        """bool: whether the detector has img head"""
        return hasattr(self, 'img_head') and self.img_head is not None

    def init_extra_weights(self):
        if self.with_img_head:
            self.img_head.init_weights()
        if self.with_patch_head:
            self.patch_head.init_weights()

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

    def forward_patch_head(self, imgs1, imgs2, x1, x2, clip_len):
        patch_bboxes1, _ = get_random_crop_bbox(
            imgs1.size(0) * self.patch_num,
            self.patch_size,
            imgs1.shape[2:],
            device=imgs1.device,
            center_ratio=0,
            border=0)
        patch_bboxes2, _ = get_random_crop_bbox(
            imgs2.size(0) * self.patch_num,
            self.patch_size,
            imgs2.shape[2:],
            device=imgs2.device,
            center_ratio=0,
            border=0)
        if self.patch_num > 1:
            x1 = x1.expand(self.patch_num, -1, -1, -1,
                           -1).reshape(-1, *x1.shape[1:])
            x2 = x2.expand(self.patch_num, -1, -1, -1,
                           -1).reshape(-1, *x2.shape[1:])
            patch_bboxes1 = bbox2roi(
                patch_bboxes1.view(imgs1.size(0), self.patch_num, 4))
            patch_bboxes2 = bbox2roi(
                patch_bboxes2.view(imgs2.size(0), self.patch_num, 4))
            if self.patch_from_img:
                patch_x1 = self.extract_feat(
                    self.roi_align(imgs1, patch_bboxes1))
                patch_x2 = self.extract_feat(
                    self.roi_align(imgs2, patch_bboxes2))
            else:
                patch_x1 = self.roi_align(x1, patch_bboxes1)
                patch_x2 = self.roi_align(x2, patch_bboxes2)
        else:
            if self.patch_from_img:
                patch_x1 = self.extract_feat(
                    crop_and_resize(imgs1, patch_bboxes1, self.patch_size))
                patch_x2 = self.extract_feat(
                    crop_and_resize(imgs2, patch_bboxes2, self.patch_size))
            else:
                stride = imgs1.size(2) // x1.size(2)
                patch_x1 = crop_and_resize(
                    x1, patch_bboxes1 / stride,
                    tuple_divide(self.patch_size, stride))
                patch_x2 = crop_and_resize(
                    x2, patch_bboxes2 / stride,
                    tuple_divide(self.patch_size, stride))
        if self.patch_tracking:
            if self.patch_mask_radius is not None:
                mask = spatial_neighbor(
                    x1.size(0),
                    x1.size(2),
                    x2.size(3),
                    neighbor_range=self.patch_mask_radius,
                    device=x1.device,
                    dtype=x1.dtype).view(
                        x1.size(2), x1.size(3), x1.size(2), x1.size(3))
                mask = resize_spatial_mask(mask, patch_x1.shape[2:]).view(
                    x1.shape[2:].numel(), patch_x1.shape[2:].numel())
            else:
                mask = None
            patch_x12 = masked_attention_efficient(
                patch_x1, x2, x2, mask, mode=self.patch_att_mode)
            if self.patch_cycle_tracking:
                patch_x12 = masked_attention_efficient(
                    patch_x12, x1, x1, mask, mode=self.patch_att_mode)
            patch_x21 = masked_attention_efficient(
                patch_x2, x1, x1, mask, mode=self.patch_att_mode)
            if self.patch_cycle_tracking:
                patch_x21 = masked_attention_efficient(
                    patch_x12, x2, x2, mask, mode=self.patch_att_mode)
        else:
            # pseudo patch x12, x21
            patch_x12 = patch_x2
            patch_x21 = patch_x1

        loss_weight = 1. / clip_len if self.intra_video else 1.
        losses = dict()
        z1, p1 = self.patch_head(patch_x1)
        z2, p2 = self.patch_head(patch_x12)
        losses.update(
            add_prefix(
                self.patch_head.loss(p1, z1, p2, z2, weight=loss_weight),
                prefix='0.0'))
        if self.intra_video:
            z2_v, p2_v = images2video(z2, clip_len), images2video(p2, clip_len)
            for i in range(1, clip_len):
                losses.update(
                    add_prefix(
                        self.patch_head.loss(
                            p1,
                            z1,
                            video2images(p2_v.roll(i, dims=2)),
                            video2images(z2_v.roll(i, dims=2)),
                            weight=loss_weight),
                        prefix=f'0.{i}'))
        if self.patch_tracking:
            z1, p1 = self.patch_head(patch_x21)
            z2, p2 = self.patch_head(patch_x2)
            losses.update(
                add_prefix(
                    self.patch_head.loss(p1, z1, p2, z2, weight=loss_weight),
                    prefix='1.0'))
            if self.intra_video:
                z2_v, p2_v = images2video(z2, clip_len), images2video(
                    p2, clip_len)
                for i in range(1, clip_len):
                    losses.update(
                        add_prefix(
                            self.patch_head.loss(
                                p1,
                                z1,
                                video2images(p2_v.roll(i, dims=2)),
                                video2images(z2_v.roll(i, dims=2)),
                                weight=loss_weight),
                            prefix=f'1.{i}'))
        if self.with_cls_head and self.cls_on_patch:
            losses.update(
                add_prefix(
                    self.forward_cls_head(patch_x1, patch_x12, clip_len),
                    prefix='cls.0'))
        if self.with_cls_head and self.cls_on_patch and self.patch_tracking:
            losses.update(
                add_prefix(
                    self.forward_cls_head(patch_x21, patch_x2, clip_len),
                    prefix='cls.1'))

        return losses

    def forward_train(self, imgs, labels=None):
        # [B, N, C, T, H, W]
        assert imgs.size(1) == 2
        clip_len = imgs.size(3)
        imgs1 = video2images(imgs[:,
                                  0].contiguous().reshape(-1, *imgs.shape[2:]))
        imgs2 = video2images(imgs[:,
                                  1].contiguous().reshape(-1, *imgs.shape[2:]))
        x1 = self.backbone(imgs1)
        x2 = self.backbone(imgs2)
        # x_cat = self.backbone(cat([imgs1, imgs2]))
        # x1, x2 = x_cat.split(imgs1.size(0))
        losses = dict()
        if self.with_img_head:
            loss_img_head = self.forward_img_head(x1, x2, clip_len)
            losses.update(add_prefix(loss_img_head, prefix='img_head'))
        if self.with_patch_head:
            loss_patch_head = self.forward_patch_head(imgs1, imgs2,
                                                      self.neck(x1),
                                                      self.neck(x2), clip_len)
            losses.update(add_prefix(loss_patch_head, prefix='patch_head'))
        return losses
