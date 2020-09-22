import kornia.augmentation as K
import torch.nn as nn
from torch.nn.modules.utils import _pair

from ..common import (crop_and_resize, get_crop_grid, get_random_crop_bbox,
                      images2video, video2images)
from ..registry import WALKERS
from .vanilla_tracker import VanillaTracker


@WALKERS.register_module()
class UVCTracker(VanillaTracker):
    """3D recognizer model framework."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stride = self.backbone.output_stride
        if self.train_cfg is not None:
            self.patch_img_size = _pair(self.train_cfg.patch_size)
            self.patch_x_size = _pair(self.train_cfg.patch_size // self.stride)
            if self.train_cfg.get('strong_aug', False):
                self.aug = nn.Sequential(
                    K.RandomRotation(degrees=10),
                    # K.RandomResizedCrop(size=self.patch_img_size,
                    #                     scale=(0.7, 0.9),
                    #                     ratio=(0.7, 1.3)),
                    K.ColorJitter(
                        brightness=0.2, contrast=0.3, saturation=0.3, hue=0.1))
            else:
                self.aug = nn.Identity()

    def crop_x_from_img(self, img, x, bboxes, crop_first):
        if crop_first:
            crop_x = self.extract_feat(
                crop_and_resize(img, bboxes * self.stride,
                                self.patch_img_size))
        else:
            crop_x = crop_and_resize(x, bboxes, self.patch_x_size)

        return crop_x

    def forward_train(self, imgs, labels=None):
        """Defines the computation performed at every call when training."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        batches, clip_len = imgs.size(0), imgs.size(2)
        assert clip_len == 2
        x = images2video(
            self.extract_feat(self.aug(video2images(imgs))), clip_len)
        ref_frame = imgs[:, :, 0].contiguous()
        tar_frame = imgs[:, :, 1].contiguous()
        ref_x = x[:, :, 0].contiguous()
        tar_x = x[:, :, 1].contiguous()

        # all bboxes are in feature space
        ref_crop_bboxes, is_center_crop = get_random_crop_bbox(
            batches,
            self.patch_x_size,
            ref_x.shape[2:],
            device=x.device,
            center_ratio=self.train_cfg.center_ratio)
        ref_crop_x = self.crop_x_from_img(ref_frame, ref_x, ref_crop_bboxes,
                                          self.train_cfg.img_as_ref)
        ref_crop_grid = get_crop_grid(ref_frame, ref_crop_bboxes * self.stride,
                                      self.patch_img_size)
        if is_center_crop:
            tar_bboxes = ref_crop_bboxes
        else:
            tar_bboxes = self.cls_head.get_tar_bboxes(ref_crop_x, tar_x)

        tar_crop_x = self.crop_x_from_img(tar_frame, tar_x, tar_bboxes,
                                          self.train_cfg.img_as_tar)

        if is_center_crop:
            ref_pred_bboxes = ref_crop_bboxes
        else:
            ref_pred_bboxes = self.cls_head.get_tar_bboxes(tar_crop_x, ref_x)

        ref_pred_crop_x = self.crop_x_from_img(ref_frame, ref_x,
                                               ref_pred_bboxes,
                                               self.train_cfg.img_as_ref_pred)

        ref_pred_crop_grid = get_crop_grid(ref_frame,
                                           ref_pred_bboxes * self.stride,
                                           self.patch_img_size)

        loss = dict()

        loss.update(self.cls_head.loss(ref_crop_x, tar_crop_x, 'ref_tar'))
        loss.update(self.cls_head.loss(tar_crop_x, ref_pred_crop_x, 'tar_ref'))
        loss['dist_bbox'] = self.cls_head.loss_bbox(
            ref_pred_bboxes / self.patch_x_size[0],
            ref_crop_bboxes / self.patch_x_size[0])
        loss['loss_bbox'] = self.cls_head.loss_bbox(ref_crop_grid,
                                                    ref_pred_crop_grid)

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
