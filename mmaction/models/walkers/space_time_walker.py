import kornia.augmentation as K
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import pil_nearest_interpolate
from ..registry import WALKERS
from .base import BaseWalker


@WALKERS.register_module()
class SpaceTimeWalker(BaseWalker):
    """3D recognizer model framework."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.train_cfg is not None:
            patch_size = self.train_cfg.patch_size
            patch_stride = self.train_cfg.patch_stride
            self.unfold = nn.Unfold((patch_size, patch_size),
                                    stride=(patch_stride, patch_stride))
            self.spatial_jitter = K.RandomResizedCrop(
                size=(patch_size, patch_size),
                scale=(0.7, 0.9),
                ratio=(0.7, 1.3))

    def video2patch(self, x):
        """
        Args:
            x: Tensor of shape (N,C,T,H,W).
        """
        patch_size = self.train_cfg.patch_size
        batch, channels, depth, height, width = x.shape
        # [N * T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
        # input_x = x
        # [N * T, C x h x w, P]
        x = self.unfold(x)
        # [N * T * P, C, h, w]
        x = x.permute(0, 2, 1).reshape(-1, channels, patch_size, patch_size)
        # assert torch.allclose(input_x[:, :, :patch_size, :patch_size],
        #                       x.view(batch * depth, -1, channels,
        #                              patch_size, patch_size)[:, 0])
        x = self.spatial_jitter(x)

        return x

    def forward_train(self, imgs, labels=None):
        """Defines the computation performed at every call when training."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        batches, clip_len = imgs.size(0), imgs.size(2)
        patches = self.video2patch(imgs)

        x = self.extract_feat(patches)
        cls_score = self.cls_head(x, batches, clip_len)
        loss = self.cls_head.loss(cls_score)

        return loss

    def compute_affinity(self,
                         imgs,
                         src_idx,
                         dst_idx,
                         temperature=1.,
                         normalize=True):
        src_feat = self.extract_feat(imgs[:, :, src_idx])
        dst_feat = self.extract_feat(imgs[:, :, dst_idx])
        batches, channels, height, width = src_feat.size()
        src_feat = src_feat.view(batches, channels, height * width)
        dst_feat = dst_feat.view(batches, channels, height * width)
        if normalize:
            src_feat = F.normalize(src_feat, p=2, dim=1)
            dst_feat = F.normalize(dst_feat, p=2, dim=1)
        src_feat = src_feat.permute(0, 2, 1).contiguous()
        dst_feat = dst_feat.contiguous()
        affinity = torch.bmm(src_feat, dst_feat) / temperature
        affinity = affinity.softmax(dim=1)

        return affinity

    def propagate(self, pred, affinity, topk):
        batches, channels, height, width = pred.size()
        tk_val, tk_idx = affinity.topk(dim=1, k=topk)
        tk_val_min, _ = tk_val.min(dim=1)
        tk_val_min = tk_val_min.view(batches, 1, height * width)
        affinity[tk_val_min > affinity] = 0
        pred = pred.view(batches, channels, -1)
        new_pred = torch.bmm(pred, affinity)
        return new_pred.view(batches, channels, height, width)

    def forward_test(self, imgs, ref_seg_map, img_meta):
        """Defines the computation performed at every call when evaluation and
        testing."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        # get target shape
        feat_shape = self.extract_feat(imgs[0:1, :, 0]).shape
        resized_seg_map = pil_nearest_interpolate(
            ref_seg_map.unsqueeze(1), size=feat_shape[2:]).squeeze(1).long()
        resized_seg_map = F.one_hot(resized_seg_map).permute(0, 3, 1,
                                                             2).float()
        clip_len = imgs.size(2)
        idx_bank = []
        seg_bank = []

        ref_seg_map = F.interpolate(
            ref_seg_map.unsqueeze(1),
            size=img_meta[0]['original_shape'][:2],
            mode='nearest').squeeze(1)

        seg_preds = [ref_seg_map.detach().cpu().numpy()]

        for frame_idx in range(1, clip_len):
            affinity = self.compute_affinity(
                imgs, 0, frame_idx, temperature=self.test_cfg.temperature)
            seg_logit = self.propagate(resized_seg_map, affinity,
                                       self.test_cfg.topk)
            assert len(idx_bank) == len(seg_bank)
            for hist_idx, hist_seg in zip(idx_bank, seg_bank):
                hist_affinity = self.compute_affinity(
                    imgs,
                    hist_idx,
                    frame_idx,
                    temperature=self.test_cfg.temperature)
                seg_logit += self.propagate(hist_seg, hist_affinity,
                                            self.test_cfg.topk)
            seg_logit /= 1 + len(idx_bank)

            idx_bank.append(frame_idx)
            seg_bank.append(seg_logit)

            if len(idx_bank) > self.test_cfg.precede_frames:
                idx_bank.pop(0)
            if len(seg_bank) > self.test_cfg.precede_frames:
                seg_bank.pop(0)
            seg_pred = F.interpolate(
                seg_logit,
                size=img_meta[0]['original_shape'][:2],
                mode='bilinear',
                align_corners=False)
            seg_pred_min = seg_pred.view(*seg_pred.shape[:2], -1).min(
                dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
            seg_pred_max = seg_pred.view(*seg_pred.shape[:2], -1).max(
                dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
            normalized_seg_pred = (seg_pred - seg_pred_min) / (
                seg_pred_max - seg_pred_min)
            seg_pred = torch.where(seg_pred_max > 0, normalized_seg_pred,
                                   seg_pred)
            seg_pred = seg_pred.argmax(dim=1)
            seg_pred = F.interpolate(
                seg_pred.byte().unsqueeze(1),
                size=img_meta[0]['original_shape'][:2],
                mode='nearest').squeeze(1)
            seg_preds.append(seg_pred.detach().cpu().numpy())

        seg_preds = np.stack(seg_preds, axis=1)
        # unravel batch dim
        return list(seg_preds)

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

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        if mode:
            self.backbone.switch_strides()
            self.backbone.switch_out_indices()
        else:
            self.backbone.switch_strides(self.test_cfg.strides)
            self.backbone.switch_out_indices(self.test_cfg.out_indices)
