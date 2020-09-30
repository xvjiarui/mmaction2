import kornia.augmentation as K
import numpy.random as npr
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmaction.utils import add_suffix
from ..common import (crop_and_resize, get_crop_grid, get_random_crop_bbox,
                      get_top_diff_crop_bbox, images2video, video2images)
from ..registry import WALKERS
from .vanilla_tracker import VanillaTracker


@WALKERS.register_module()
class UVCTrackerRecursive(VanillaTracker):
    """3D recognizer model framework."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stride = self.backbone.output_stride
        if self.train_cfg is not None:
            self.patch_img_size = _pair(self.train_cfg.patch_size)
            self.patch_x_size = _pair(self.train_cfg.patch_size // self.stride)
            if self.train_cfg.get('strong_aug', False):
                same_on_batch = self.train_cfg.get('same_on_batch', False)
                self.aug = nn.Sequential(
                    K.RandomRotation(degrees=10, same_on_batch=same_on_batch),
                    # K.RandomResizedCrop(size=self.patch_img_size,
                    #                     scale=(0.7, 0.9),
                    #                     ratio=(0.7, 1.3)),
                    K.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.1,
                        same_on_batch=same_on_batch))
            else:
                self.aug = nn.Identity()
            self.border = self.train_cfg.get('border', 0)
            self.grid_size = self.train_cfg.get('grid_size', 9)
            self.img_as_ref = self.train_cfg.get('img_as_ref')
            self.img_as_tar = self.train_cfg.get('img_as_tar')
            self.diff_crop = self.train_cfg.get('diff_crop', False)
            self.img_as_grid = self.train_cfg.get('img_as_grid', True)
            self.recursive_times = self.train_cfg.get('recursive_times')
            assert self.recursive_times >= 2

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

    def track(self, tar_frame, tar_x, ref_crop_x):
        tar_bboxes = self.cls_head.get_tar_bboxes(ref_crop_x, tar_x)
        tar_crop_x = self.crop_x_from_img(tar_frame, tar_x, tar_bboxes,
                                          self.train_cfg.img_as_tar)

        return tar_bboxes, tar_crop_x

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
        assert clip_len == 2, f'{clip_len} != 2'
        ref_frame = imgs[:, :, 0]
        ref_x = x[:, :, 0]
        tar_frame = imgs[:, :, 1]
        tar_x = x[:, :, 1]
        # TODO: all bboxes are in feature space
        ref_bboxes, _ = self.get_ref_crop_bbox(batches, imgs)
        ref_crop_x = self.crop_x_from_img(ref_frame, ref_x, ref_bboxes,
                                          self.img_as_ref)
        ref_hist = []
        tar_hist = []
        ref_hist.append((ref_bboxes, ref_crop_x))
        for step in range(1, self.recursive_times):
            ref_bboxes, ref_crop_x = ref_hist[-1]
            tar_bboxes, tar_crop_x = self.track(tar_frame, tar_x, ref_crop_x)
            tar_hist.append((tar_bboxes, tar_crop_x))
            ref_bboxes, ref_crop_x = self.track(ref_frame, ref_x, tar_crop_x)
            ref_hist.append((ref_bboxes, ref_crop_x))
        ref_bboxes, ref_crop_x = ref_hist[-1]
        tar_bboxes, tar_crop_x = self.track(tar_frame, tar_x, ref_crop_x)
        tar_hist.append((tar_bboxes, tar_crop_x))

        assert len(ref_hist) == self.recursive_times
        assert len(tar_hist) == self.recursive_times

        for step in range(1, self.recursive_times):
            ref_bboxes, ref_crop_x = ref_hist[step - 1]
            ref_pred_bboxes, ref_pred_crop_x = ref_hist[step]
            ref_crop_grid = self.get_grid(ref_frame, ref_x, ref_bboxes)
            ref_pred_crop_grid = self.get_grid(ref_frame, ref_x,
                                               ref_pred_bboxes)
            loss_ref_step = dict()

            loss_ref_step['dist_bbox'] = self.cls_head.loss_bbox(
                ref_pred_bboxes / self.patch_x_size[0],
                ref_bboxes / self.patch_x_size[0])
            loss_ref_step['loss_bbox'] = self.cls_head.loss_bbox(
                ref_crop_grid, ref_pred_crop_grid)

            tar_bboxes, tar_crop_x = tar_hist[step - 1]
            tar_pred_bboxes, tar_pred_crop_x = tar_hist[step]
            tar_crop_grid = self.get_grid(tar_frame, tar_x, tar_bboxes)
            tar_pred_crop_grid = self.get_grid(tar_frame, tar_x,
                                               tar_pred_bboxes)
            loss_tar_step = dict()

            loss_tar_step['dist_bbox'] = self.cls_head.loss_bbox(
                tar_pred_bboxes / self.patch_x_size[0],
                tar_bboxes / self.patch_x_size[0])
            loss_tar_step['loss_bbox'] = self.cls_head.loss_bbox(
                tar_crop_grid, tar_pred_crop_grid)
            #
            loss_ref_step.update(
                add_suffix(
                    self.cls_head.loss(ref_crop_x, tar_crop_x),
                    suffix=f't{step}'))
            loss_tar_step.update(
                add_suffix(
                    self.cls_head.loss(tar_crop_x, ref_pred_crop_x),
                    suffix=f't{step}'))

            loss.update(add_suffix(loss_ref_step, f'ref_step{step}'))
            loss.update(add_suffix(loss_tar_step, f'tar_step{step}'))

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
