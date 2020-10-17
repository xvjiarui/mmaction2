import kornia.augmentation as K
import numpy.random as npr
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmaction.utils import add_suffix
from ..common import (bbox_overlaps, crop_and_resize, get_crop_grid,
                      get_random_crop_bbox, get_top_diff_crop_bbox,
                      images2video, video2images)
from ..registry import TRACKERS
from .vanilla_tracker import VanillaTracker


@TRACKERS.register_module()
class UVCTrackerV3(VanillaTracker):
    """3D recognizer model framework."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stride = self.backbone.output_stride
        if self.train_cfg is not None:
            self.patch_img_size = _pair(self.train_cfg.patch_size)
            self.patch_x_size = _pair(self.train_cfg.patch_size // self.stride)
            if self.train_cfg.get('strong_aug', False):
                same_on_batch = self.train_cfg.get('same_on_batch', False)
                self.aug = K.RandomRotation(
                    degrees=10, same_on_batch=same_on_batch)
            else:
                self.aug = nn.Identity()
            self.skip_cycle = self.train_cfg.get('skip_cycle', False)
            self.border = self.train_cfg.get('border', 0)
            self.grid_size = self.train_cfg.get('grid_size', 9)
            self.img_as_ref = self.train_cfg.get('img_as_ref')
            self.img_as_tar = self.train_cfg.get('img_as_tar')
            self.diff_crop = self.train_cfg.get('diff_crop', False)
            self.img_as_grid = self.train_cfg.get('img_as_grid', True)
            self.loss_on_grid = self.train_cfg.get('loss_on_grid', True)

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

    def get_grid(self, frame, x, bboxes):
        if self.img_as_grid:
            crop_grid = get_crop_grid(frame, bboxes * self.stride,
                                      self.patch_img_size)
        else:
            crop_grid = get_crop_grid(x, bboxes, self.patch_x_size)

        return crop_grid

    def get_ref_crop_bbox(self, batches, imgs):
        if self.diff_crop:
            is_center_crop = False
            ref_bboxes = get_top_diff_crop_bbox(
                imgs[:, :, 0],
                imgs[:, :, -1],
                self.patch_img_size,
                self.grid_size,
                device=imgs.device)
        else:
            ref_bboxes, is_center_crop = get_random_crop_bbox(
                batches,
                self.patch_img_size,
                imgs.shape[2:],
                device=imgs.device,
                center_ratio=self.train_cfg.center_ratio,
                border=self.border)
        return ref_bboxes / self.stride, is_center_crop

    def forward_train(self, imgs, labels=None):
        """Defines the computation performed at every call when training."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        batches, clip_len = imgs.size(0), imgs.size(2)
        x = images2video(
            self.extract_feat(self.aug(video2images(imgs))), clip_len)
        loss = dict()
        for step in range(2, clip_len + 1):
            # step_weight = 1. if step == 2 or self.iteration > 1000 else 0
            ref_frame = imgs[:, :, 0].contiguous()
            ref_x = x[:, :, 0].contiguous()
            ref_bboxes, _ = self.get_ref_crop_bbox(batches, imgs)
            ref_crop_x = self.crop_x_from_img(ref_frame, ref_x, ref_bboxes,
                                              self.img_as_ref)
            # TODO: all bboxes are in feature space
            ref_crop_grid = self.get_grid(ref_frame, ref_x, ref_bboxes)

            tar_frame = imgs[:, :, step - 1].contiguous()
            tar_x = x[:, :, step - 1].contiguous()

            step_x = x[:, :, :step].contiguous()
            tar_bboxes = self.cls_head.get_tar_bboxes(ref_crop_x, step_x,
                                                      ref_bboxes)
            tar_crop_x = self.crop_x_from_img(tar_frame, tar_x, tar_bboxes,
                                              self.img_as_tar)
            ref_pred_bboxes = self.cls_head.get_tar_bboxes(
                tar_crop_x, step_x.flip(dims=(2, )), tar_bboxes)
            ref_pred_crop_x = self.crop_x_from_img(ref_frame, ref_x,
                                                   ref_pred_bboxes,
                                                   self.img_as_tar)
            ref_pred_crop_grid = self.get_grid(ref_frame, ref_x,
                                               ref_pred_bboxes)
            loss_step = dict()
            loss_step['iou_bbox'] = bbox_overlaps(
                ref_pred_bboxes, ref_bboxes, is_aligned=True)
            if self.loss_on_grid:
                loss_step['loss_bbox'] = self.cls_head.loss_bbox(
                    ref_crop_grid, ref_pred_crop_grid)
            else:
                loss_step['loss_bbox'] = self.cls_head.loss_bbox(
                    ref_pred_bboxes, ref_bboxes)

            loss_step.update(
                add_suffix(
                    self.cls_head.loss(ref_crop_x, tar_crop_x),
                    suffix='forward'))
            loss_step.update(
                add_suffix(
                    self.cls_head.loss(tar_crop_x, ref_pred_crop_x),
                    suffix='backward'))
            loss.update(add_suffix(loss_step, f'step{step}'))

            if self.skip_cycle and step > 2:
                loss_skip = dict()
                tar_bboxes = self.cls_head.get_tar_bboxes(
                    ref_crop_x, tar_x, ref_bboxes)
                tar_crop_x = self.crop_x_from_img(tar_frame, tar_x, tar_bboxes,
                                                  self.img_as_tar)
                ref_pred_bboxes = self.cls_head.get_tar_bboxes(
                    tar_crop_x, ref_x, tar_bboxes)
                ref_pred_crop_x = self.crop_x_from_img(ref_frame, ref_x,
                                                       ref_pred_bboxes,
                                                       self.img_as_tar)
                ref_pred_crop_grid = self.get_grid(ref_frame, ref_x,
                                                   ref_pred_bboxes)
                loss_skip['iou_bbox'] = bbox_overlaps(
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
