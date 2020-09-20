from torch.nn.modules.utils import _pair

from ..common import (crop_and_resize, get_random_crop_bbox, images2video,
                      video2images)
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
            # if self.train_cfg.img_as_ref:
            #     patch_size = self.train_cfg.patch_size
            # else:
            #     patch_size = self.train_cfg.patch_size // self.stride
            # degrees = self.train_cfg.degrees
            # self.aug_crop = nn.Sequential(
            #     K.RandomRotation(degrees=degrees),
            #     K.RandomCrop(size=(patch_size, patch_size)))
            # self.center_crop = K.CenterCrop(size=(self.train_cfg.patch_size,
            #                                       self.train_cfg.patch_size))

    def forward_train(self, imgs, labels=None):
        """Defines the computation performed at every call when training."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        batches, clip_len = imgs.size(0), imgs.size(2)
        assert clip_len == 2
        ref_frame = imgs[:, :, 0]
        tar_frame = imgs[:, :, 1]
        x = images2video(self.extract_feat(video2images(imgs)), clip_len)
        ref_x = x[:, :, 0]
        tar_x = x[:, :, 1]

        # all bboxes are in feature space
        ref_crop_bboxes = get_random_crop_bbox(
            batches, self.patch_x_size, ref_x.shape[2:], device=x.device)
        if self.train_cfg.img_as_ref:
            # ref_crop_x = self.extract_feat(self.aug_crop(ref_frame))
            ref_crop_x = self.extract_feat(
                crop_and_resize(ref_frame, ref_crop_bboxes * self.stride,
                                self.patch_img_size))
        else:
            # ref_crop_x = self.aug_crop(ref_x)
            ref_crop_x = crop_and_resize(ref_x, ref_crop_bboxes,
                                         self.patch_x_size)
        tar_bboxes = self.cls_head.get_tar_bboxes(ref_crop_x, tar_x)

        if self.train_cfg.img_as_tar:
            tar_crop_x = self.extract_feat(
                crop_and_resize(tar_frame, tar_bboxes * self.stride,
                                self.patch_img_size))
        else:
            tar_crop_x = crop_and_resize(tar_x, tar_bboxes, self.patch_x_size)
        ref_pred_bboxes = self.cls_head.get_tar_bboxes(tar_crop_x, ref_x)

        if self.train_cfg.img_as_ref_pred:
            ref_pred_crop_x = self.extract_feat(
                crop_and_resize(ref_frame, ref_pred_bboxes * self.stride,
                                self.patch_img_size))
        else:
            ref_pred_crop_x = crop_and_resize(ref_x, ref_pred_bboxes,
                                              self.patch_x_size)

        loss = dict()

        loss.update(self.cls_head.loss(ref_crop_x, tar_crop_x, 'ref_tar'))
        loss.update(self.cls_head.loss(tar_crop_x, ref_pred_crop_x, 'tar_ref'))
        # loss['loss_bbox'] = self.cls_head.loss_bbox(
        #     ref_pred_bboxes/self.patch_x_size[0],
        #     ref_crop_bboxes/self.patch_x_size[0])
        # loss.update(self.cls_head.loss(
        #     self.extract_feat(self.center_crop(ref_frame)),
        #     self.extract_feat(self.center_crop(tar_frame)), 'center'))

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
