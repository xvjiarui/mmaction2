import kornia.augmentation as K
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmaction.utils import add_suffix
from ..common import (crop_and_resize, get_crop_grid, get_random_crop_bbox,
                      images2video, video2images)
from ..registry import WALKERS
from .vanilla_tracker import VanillaTracker


@WALKERS.register_module()
class UVCTrackerV2(VanillaTracker):
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
            self.skip_cycle = self.train_cfg.get('skip_cycle', False)

    def crop_x_from_img(self, img, x, bboxes, crop_first):
        if crop_first:
            crop_x = self.extract_feat(
                crop_and_resize(img, bboxes * self.stride,
                                self.patch_img_size))
        else:
            crop_x = crop_and_resize(x, bboxes, self.patch_x_size)

        return crop_x

    def track(self, tar_frame, tar_x, ref_crop_x, tar_bboxes=None):
        if tar_bboxes is None:
            tar_bboxes = self.cls_head.get_tar_bboxes(ref_crop_x, tar_x)
        tar_crop_x = self.crop_x_from_img(tar_frame, tar_x, tar_bboxes,
                                          self.train_cfg.img_as_tar)

        return tar_bboxes, tar_crop_x

    def forward_train(self, imgs, labels=None):
        """Defines the computation performed at every call when training."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        batches, clip_len = imgs.size(0), imgs.size(2)
        x = images2video(
            self.extract_feat(self.aug(video2images(imgs))), clip_len)
        loss = dict()
        for step in range(2, clip_len + 1):
            # step_weight = 1. if step == 2 or self.iteration > 1000 else 0
            step_weight = 1.
            skip_weight = 1.
            ref_frame = imgs[:, :, 0]
            ref_x = x[:, :, 0]
            # all bboxes are in feature space
            ref_bboxes, is_center_crop = get_random_crop_bbox(
                batches,
                self.patch_x_size,
                ref_x.shape[2:],
                device=x.device,
                center_ratio=self.train_cfg.center_ratio)
            ref_crop_x = self.crop_x_from_img(ref_frame, ref_x, ref_bboxes,
                                              self.train_cfg.img_as_ref)
            ref_crop_grid = get_crop_grid(ref_frame, ref_bboxes * self.stride,
                                          self.patch_img_size)
            forward_hist = [(ref_bboxes, ref_crop_x)]
            for tar_idx in range(1, step):
                last_bboxes, last_crop_x = forward_hist[-1]
                tar_frame = imgs[:, :, tar_idx]
                tar_x = x[:, :, tar_idx]
                tar_bboxes, tar_crop_x = self.track(
                    tar_frame,
                    tar_x,
                    last_crop_x,
                    tar_bboxes=ref_bboxes if is_center_crop else None)
                forward_hist.append((tar_bboxes, tar_crop_x))
            assert len(forward_hist) == step

            backward_hist = [forward_hist[-1]]
            for last_idx in reversed(range(1, step)):
                tar_idx = last_idx - 1
                last_bboxes, last_crop_x = backward_hist[-1]
                tar_frame = imgs[:, :, tar_idx]
                tar_x = x[:, :, tar_idx]
                tar_bboxes, tar_crop_x = self.track(
                    tar_frame,
                    tar_x,
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
                # loss.update(
                #     add_suffix(
                #         self.cls_head.loss(last_crop_x, tar_crop_x,
                #                            'forward', weight=step_weight),
                #         f'step{step}.t{tar_idx}'))
            for last_idx in reversed(range(1, step)):
                tar_crop_x = backward_hist[last_idx - 1][1]
                last_crop_x = backward_hist[last_idx][1]
                loss_step.update(
                    add_suffix(
                        self.cls_head.loss(
                            tar_crop_x, last_crop_x, weight=step_weight),
                        suffix=f'backward.t{last_idx}'))
                # loss.update(
                #     add_suffix(
                #         self.cls_head.loss(
                #             tar_crop_x, last_crop_x, 'backward',
                #             weight=step_weight), f'step{step}.t{last_idx}'))
            loss.update(add_suffix(loss_step, f'step{step}'))
            if self.skip_cycle and step > 2:
                loss_skip = dict()
                tar_frame = imgs[:, :, step - 1]
                tar_x = x[:, :, step - 1]
                _, tar_crop_x = self.track(
                    tar_frame,
                    tar_x,
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
                # loss.update(
                #     add_suffix(
                #         self.cls_head.loss(ref_crop_x, tar_crop_x,
                #                            'forward', weight=step_weight),
                #         f'skip{step}'))
                # loss.update(
                #     add_suffix(
                #         self.cls_head.loss(tar_crop_x, ref_pred_crop_x,
                #                            'backward', weight=step_weight),
                #         f'skip{step}'))

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
