from torch.nn.modules.utils import _pair

from mmaction.utils import add_prefix, tuple_divide
from .. import builder
from ..common import (bbox_overlaps, crop_and_resize, get_crop_grid,
                      get_random_crop_bbox, images2video, video2images)
from ..registry import TRACKERS
from .vanilla_tracker import VanillaTracker


@TRACKERS.register_module()
class SimSiamUVCTracker(VanillaTracker):
    """3D recognizer model framework."""

    def __init__(self,
                 *args,
                 backbone,
                 patch_head=None,
                 img_head=None,
                 track_head=None,
                 **kwargs):
        super().__init__(*args, backbone=backbone, **kwargs)
        if patch_head is not None:
            self.patch_head = builder.build_head(patch_head)
        if img_head is not None:
            self.img_head = builder.build_head(img_head)
        if track_head is not None:
            self.track_head = builder.build_head(track_head)
        self.init_extra_weights()
        if self.train_cfg is not None:
            self.shared_neck = self.train_cfg.get('shared_neck', False)
            self.intra_video = self.train_cfg.get('intra_video', False)
            self.patch_size = _pair(self.train_cfg.get('patch_size', 96))
            self.patch_from_img = self.train_cfg.get('patch_from_img', False)
            self.cls_on_patch = self.train_cfg.get('cls_on_patch', False)
            self.cls_on_neck = self.train_cfg.get('cls_on_neck', True)
            self.img_on_patch = self.train_cfg.get('img_on_patch', False)
            self.implicit_cycle = self.train_cfg.get('implicit_cycle', False)

    @property
    def with_patch_head(self):
        """bool: whether the detector has patch head"""
        return hasattr(self, 'patch_head') and self.patch_head is not None

    @property
    def with_img_head(self):
        """bool: whether the detector has img head"""
        return hasattr(self, 'img_head') and self.img_head is not None

    @property
    def with_track_head(self):
        """bool: whether the detector has track head"""
        return hasattr(self, 'track_head') and self.track_head is not None

    def init_extra_weights(self):
        if self.with_img_head:
            self.img_head.init_weights()
        if self.with_patch_head:
            self.patch_head.init_weights()
        if self.with_track_head:
            self.track_head.init_weights()

    def forward_neck(self, x):
        if self.shared_neck:
            neck_x = self.neck(x, self.backbone)
        else:
            neck_x = self.neck(x)

        return neck_x

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

    def forward_patch_head(self, x1, x2, clip_len):
        losses = dict()
        z1, p1 = self.patch_head(x1)
        z2, p2 = self.patch_head(x2)
        loss_weight = 1. / clip_len if self.intra_video else 1.
        losses.update(
            add_prefix(
                self.patch_head.loss(p1, z1, p2, z2, weight=loss_weight),
                prefix='0'))
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
                        prefix=f'{i}'))
        return losses

    def forward_track_head(self, imgs1, imgs2, x1, x2, grids1, grids2):
        # TODO patch_bboxes are in image domain
        patch_bboxes1, _ = get_random_crop_bbox(
            imgs1.size(0),
            self.patch_size,
            imgs1.shape[2:],
            device=imgs1.device,
            center_ratio=0,
            border=0)
        patch_bboxes2, _ = get_random_crop_bbox(
            imgs2.size(0),
            self.patch_size,
            imgs2.shape[2:],
            device=imgs2.device,
            center_ratio=0,
            border=0)
        stride = imgs1.size(2) // x1.size(2)
        if self.patch_from_img:
            patch_x1 = self.extract_feat(
                crop_and_resize(imgs1, patch_bboxes1, self.patch_size))
            patch_x2 = self.extract_feat(
                crop_and_resize(imgs2, patch_bboxes2, self.patch_size))
        else:
            patch_x1 = crop_and_resize(x1, patch_bboxes1 / stride,
                                       tuple_divide(self.patch_size, stride))
            patch_x2 = crop_and_resize(x2, patch_bboxes2 / stride,
                                       tuple_divide(self.patch_size, stride))

        pred_patch_bboxes2 = self.track_head(patch_x1, x2) * stride
        pred_patch_x2 = crop_and_resize(x2, pred_patch_bboxes2 / stride,
                                        tuple_divide(self.patch_size, stride))
        pred_patch_bboxes1 = self.track_head(patch_x2, x1) * stride
        pred_patch_x1 = crop_and_resize(x1, pred_patch_bboxes1 / stride,
                                        tuple_divide(self.patch_size, stride))

        losses = dict()
        if self.implicit_cycle:
            affinity12, patch_grid1, cycle_patch_grid1 \
                = self.track_head.estimate_grid(patch_x1, pred_patch_x2)
            affinity21, patch_grid2, cycle_patch_grid2 \
                = self.track_head.estimate_grid(patch_x2, pred_patch_x1)

        else:
            # cycle tracking
            if grids1 is None:
                patch_grid1 = get_crop_grid(imgs1, patch_bboxes1,
                                            self.patch_size).permute(
                                                0, 3, 1, 2).contiguous()
            else:
                patch_grid1 = crop_and_resize(grids1, patch_bboxes1,
                                              self.patch_size)
            if grids2 is None:
                patch_grid2 = get_crop_grid(imgs2, patch_bboxes2,
                                            self.patch_size).permute(
                                                0, 3, 1, 2).contiguous()
            else:
                patch_grid2 = crop_and_resize(grids2, patch_bboxes1,
                                              self.patch_size)
            cycle_patch_bboxes1 = self.track_head(pred_patch_x2, x1) * stride
            if grids1 is None:
                cycle_patch_grid1 = get_crop_grid(imgs1, cycle_patch_bboxes1,
                                                  self.patch_size).permute(
                                                      0, 3, 1, 2).contiguous()
            else:
                cycle_patch_grid1 = crop_and_resize(grids1,
                                                    cycle_patch_bboxes1,
                                                    self.patch_size)
            cycle_patch_bboxes2 = self.track_head(pred_patch_x1, x2) * stride
            if grids2 is None:
                cycle_patch_grid2 = get_crop_grid(imgs2, cycle_patch_bboxes2,
                                                  self.patch_size).permute(
                                                      0, 3, 1, 2).contiguous()
            else:
                cycle_patch_grid2 = crop_and_resize(grids2,
                                                    cycle_patch_bboxes2,
                                                    self.patch_size)
            losses['iou.forward'] = bbox_overlaps(
                patch_bboxes1, cycle_patch_bboxes1, is_aligned=True)
            losses['iou.backward'] = bbox_overlaps(
                patch_bboxes2, cycle_patch_bboxes2, is_aligned=True)
            affinity12 = None
            affinity21 = None

        losses.update(
            add_prefix(
                self.track_head.loss(patch_grid1, cycle_patch_grid1,
                                     affinity12),
                prefix='forward'))
        losses.update(
            add_prefix(
                self.track_head.loss(patch_grid2, cycle_patch_grid2,
                                     affinity21),
                prefix='backward'))

        track_results = dict(
            patch_x1=patch_x1,
            patch_x2=patch_x2,
            pred_patch_x1=pred_patch_x1,
            pred_patch_x2=pred_patch_x2,
            patch_bboxes1=patch_bboxes1,
            patch_bboxes2=patch_bboxes2,
            pred_patch_bboxes1=pred_patch_bboxes1,
            pred_patch_bboxes2=pred_patch_bboxes2)

        return track_results, losses

    def forward_patch_track(self,
                            imgs1,
                            imgs2,
                            x1,
                            x2,
                            bx1,
                            bx2,
                            clip_len,
                            grids1=None,
                            grids2=None):
        losses = dict()

        track_results, loss_track_head = self.forward_track_head(
            imgs1, imgs2, x1, x2, grids1, grids2)

        patch_x1 = track_results['patch_x1']
        patch_x2 = track_results['patch_x2']
        pred_patch_x1 = track_results['pred_patch_x1']
        pred_patch_x2 = track_results['pred_patch_x2']
        patch_bboxes1 = track_results['patch_bboxes1']
        patch_bboxes2 = track_results['patch_bboxes2']
        pred_patch_bboxes1 = track_results['pred_patch_bboxes1']
        pred_patch_bboxes2 = track_results['pred_patch_bboxes2']

        losses.update(add_prefix(loss_track_head, prefix='track_head'))

        if self.with_patch_head:
            loss_forward_patch_head = self.forward_patch_head(
                patch_x1, pred_patch_x2, clip_len)
            losses.update(
                add_prefix(
                    loss_forward_patch_head, prefix='patch_head.forward'))
            loss_backward_patch_head = self.forward_patch_head(
                pred_patch_x1, patch_x2, clip_len)
            losses.update(
                add_prefix(
                    loss_backward_patch_head, prefix='patch_head.backward'))

        if self.with_cls_head and self.cls_on_patch:
            loss_forward_cls_head = self.forward_cls_head(
                patch_x1, pred_patch_x2, clip_len)
            losses.update(
                add_prefix(loss_forward_cls_head, prefix='cls_head.forward'))
            loss_backward_cls_head = self.forward_cls_head(
                pred_patch_x1, patch_x2, clip_len)
            losses.update(
                add_prefix(loss_backward_cls_head, prefix='cls_head.backward'))

        # TODO bx1, bx2 is backbone feature
        if self.with_img_head and self.img_on_patch:
            if isinstance(bx1, tuple):
                bx1 = bx1[-1]
            if isinstance(bx2, tuple):
                bx2 = bx2[-1]
            backbone_stride = imgs1.size(2) // bx1.size(2)
            loss_forward_img_head = self.forward_img_head(
                crop_and_resize(bx1, patch_bboxes1 / backbone_stride,
                                tuple_divide(self.patch_size,
                                             backbone_stride)),
                crop_and_resize(bx2, pred_patch_bboxes2 / backbone_stride,
                                tuple_divide(self.patch_size,
                                             backbone_stride)), clip_len)
            losses.update(
                add_prefix(loss_forward_img_head, prefix='img_head.forward'))
            loss_backward_img_head = self.forward_img_head(
                crop_and_resize(bx1, pred_patch_bboxes1 / backbone_stride,
                                tuple_divide(self.patch_size,
                                             backbone_stride)),
                crop_and_resize(bx2, patch_bboxes2 / backbone_stride,
                                tuple_divide(self.patch_size,
                                             backbone_stride)), clip_len)
            losses.update(
                add_prefix(loss_backward_img_head, prefix='img_head.backward'))

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

        assert self.with_track_head
        if self.with_patch_head or self.with_cls_head or self.with_track_head:
            neck_x1 = self.forward_neck(x1)
            neck_x2 = self.forward_neck(x2)

            if self.with_track_head or self.with_patch_head:
                loss_patch_head = self.forward_patch_track(
                    imgs1, imgs2, neck_x1, neck_x2, x1, x2, clip_len, grids1,
                    grids2)
                losses.update(add_prefix(loss_patch_head, prefix='patch_head'))

            if self.with_cls_head and self.cls_on_neck:
                loss_cls_head = self.forward_cls_head(neck_x1, neck_x2,
                                                      clip_len)
                losses.update(add_prefix(loss_cls_head, prefix='cls_head'))

        return losses
