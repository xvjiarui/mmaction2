import numpy as np
import torch
import torch.nn.functional as F

from ..common import compute_affinity, pil_nearest_interpolate, propagate
from ..registry import WALKERS
from .base import BaseWalker


@WALKERS.register_module()
class VanillaTracker(BaseWalker):
    """Pixel Tracker framework."""

    def forward_train(self, imgs, labels=None):
        raise NotImplementedError

    def forward_test(self, imgs, ref_seg_map, img_meta):
        """Defines the computation performed at every call when evaluation and
        testing."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        clip_len = imgs.size(2)
        # get target shape
        feat_shape = self.extract_feat(imgs[0:1, :, 0]).shape
        resized_seg_map = pil_nearest_interpolate(
            ref_seg_map.unsqueeze(1), size=feat_shape[2:]).squeeze(1).long()
        resized_seg_map = F.one_hot(resized_seg_map).permute(0, 3, 1,
                                                             2).float()
        idx_bank = []
        seg_bank = []

        ref_seg_map = F.interpolate(
            ref_seg_map.unsqueeze(1),
            size=img_meta[0]['original_shape'][:2],
            mode='nearest').squeeze(1)

        seg_preds = [ref_seg_map.detach().cpu().numpy()]

        for frame_idx in range(1, clip_len):
            # extract feature on-the-fly to save GPU memory
            affinity = compute_affinity(
                self.extract_feat(imgs[:, :, 0]),
                self.extract_feat(imgs[:, :, frame_idx]),
                temperature=self.test_cfg.temperature,
                softmax_dim=1)
            seg_logit = propagate(
                resized_seg_map, affinity, topk=self.test_cfg.topk)
            assert len(idx_bank) == len(seg_bank)
            for hist_idx, hist_seg in zip(idx_bank, seg_bank):
                hist_affinity = compute_affinity(
                    self.extract_feat(imgs[:, :, hist_idx]),
                    self.extract_feat(imgs[:, :, frame_idx]),
                    temperature=self.test_cfg.temperature,
                    softmax_dim=1)
                seg_logit += propagate(
                    hist_seg, hist_affinity, topk=self.test_cfg.topk)
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

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        if mode:
            self.backbone.switch_strides()
            self.backbone.switch_out_indices()
        else:
            self.backbone.switch_strides(self.test_cfg.strides)
            self.backbone.switch_out_indices(self.test_cfg.out_indices)
