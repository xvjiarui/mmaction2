import numpy as np
import torch
import torch.nn.functional as F

from ..common import (compute_affinity, pil_nearest_interpolate, propagate,
                      propagate_temporal, spatial_neighbor)
from ..registry import WALKERS
from .base import BaseTracker


@WALKERS.register_module()
class VanillaTracker(BaseTracker):
    """Pixel Tracker framework."""

    def extract_single_feat(self, imgs, idx):
        feats = self.extract_feat(imgs)
        if isinstance(feats, (tuple, list)):
            return feats[idx]
        else:
            return feats

    def forward_train(self, imgs, labels=None):
        raise NotImplementedError

    def forward_test(self, imgs, ref_seg_map, img_meta):
        """Defines the computation performed at every call when evaluation and
        testing."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        clip_len = imgs.size(2)
        # get target shape
        dummy_faet = self.extract_feat(imgs[0:1, :, 0])
        if isinstance(dummy_faet, (list, tuple)):
            feat_shapes = [_.shape for _ in dummy_faet]
        else:
            feat_shapes = [dummy_faet.shape]
        all_seg_preds = []
        for feat_idx, feat_shape in enumerate(feat_shapes):
            resized_seg_map = pil_nearest_interpolate(
                ref_seg_map.unsqueeze(1),
                size=feat_shape[2:]).squeeze(1).long()
            resized_seg_map = F.one_hot(resized_seg_map).permute(0, 3, 1,
                                                                 2).float()
            idx_bank = []
            seg_bank = []

            ref_seg_map = F.interpolate(
                ref_seg_map.unsqueeze(1),
                size=img_meta[0]['original_shape'][:2],
                mode='nearest').squeeze(1)

            seg_preds = [ref_seg_map.detach().cpu().numpy()]
            neighbor_range = self.test_cfg.get('neighbor_range', None)
            if neighbor_range is not None:
                spatial_neighbor_mask = spatial_neighbor(
                    feat_shape[0],
                    *feat_shape[2:],
                    neighbor_range=neighbor_range,
                    device=imgs.device,
                    dtype=imgs.dtype)
            else:
                spatial_neighbor_mask = None

            idx_bank.append(0)
            seg_bank.append(resized_seg_map)
            for frame_idx in range(1, clip_len):
                affinity_bank = []
                assert len(idx_bank) == len(seg_bank)
                for hist_idx, hist_seg in zip(idx_bank, seg_bank):
                    # extract feature on-the-fly to save GPU memory
                    hist_affinity = compute_affinity(
                        self.extract_single_feat(imgs[:, :, hist_idx],
                                                 feat_idx),
                        self.extract_single_feat(imgs[:, :, frame_idx],
                                                 feat_idx),
                        temperature=self.test_cfg.temperature,
                        softmax_dim=1,
                        normalize=self.test_cfg.get('with_norm', True),
                        mask=spatial_neighbor_mask)
                    # if spatial_neighbor_mask is not None:
                    #     hist_affinity *= spatial_neighbor_mask
                    #     hist_affinity = hist_affinity / hist_affinity.sum(
                    #         keepdim=True, dim=1).clamp(min=1e-12)
                    affinity_bank.append(hist_affinity)
                assert len(affinity_bank) == len(seg_bank)
                if self.test_cfg.get('with_first', True):
                    first_affinity = compute_affinity(
                        self.extract_single_feat(imgs[:, :, 0], feat_idx),
                        self.extract_single_feat(imgs[:, :, frame_idx],
                                                 feat_idx),
                        temperature=self.test_cfg.temperature,
                        softmax_dim=1,
                        normalize=self.test_cfg.get('with_norm', True),
                        mask=spatial_neighbor_mask if self.test_cfg.get(
                            'with_first_neighbor', True) else None)
                    # if (spatial_neighbor_mask is not None and
                    #         self.test_cfg.get('with_first_neighbor', True)):
                    #     first_affinity *= spatial_neighbor_mask
                    #     first_affinity = first_affinity / first_affinity.sum(
                    #         keepdim=True, dim=1).clamp(min=1e-12)
                if self.test_cfg.get('framewise', True):
                    if self.test_cfg.get('with_first', True):
                        seg_logit = propagate(
                            resized_seg_map,
                            first_affinity,
                            topk=self.test_cfg.topk)
                    else:
                        seg_logit = torch.zeros_like(resized_seg_map)
                    for hist_affinity, hist_seg in zip(affinity_bank,
                                                       seg_bank):
                        seg_logit += propagate(
                            hist_seg, hist_affinity, topk=self.test_cfg.topk)
                    seg_logit /= 1 + len(idx_bank)
                else:
                    if self.test_cfg.get('with_first', True):
                        seg_logit = propagate_temporal(
                            torch.stack([resized_seg_map] + seg_bank, dim=2),
                            torch.stack(
                                [first_affinity] + affinity_bank, dim=1),
                            topk=self.test_cfg.topk * (len(seg_bank) + 1))
                    else:
                        seg_logit = propagate_temporal(
                            torch.stack(seg_bank, dim=2),
                            torch.stack(affinity_bank, dim=1),
                            topk=self.test_cfg.topk * len(seg_bank))

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
                # seg_pred = (seg_pred - seg_pred_min) / (
                #     seg_pred_max - seg_pred_min + 1e-12)
                normalized_seg_pred = (seg_pred - seg_pred_min) / (
                    seg_pred_max - seg_pred_min + 1e-12)
                seg_pred = torch.where(seg_pred_max > 0, normalized_seg_pred,
                                       seg_pred)
                seg_pred = seg_pred.argmax(dim=1)
                seg_pred = F.interpolate(
                    seg_pred.byte().unsqueeze(1),
                    size=img_meta[0]['original_shape'][:2],
                    mode='nearest').squeeze(1)
                seg_preds.append(seg_pred.detach().cpu().numpy())

            seg_preds = np.stack(seg_preds, axis=1)
            all_seg_preds.append(seg_preds)
        if len(all_seg_preds) > 1:
            all_seg_preds = np.stack(all_seg_preds, axis=1)
        else:
            all_seg_preds = all_seg_preds[0]
        # unravel batch dim
        return list(all_seg_preds)

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        if mode:
            self.backbone.switch_strides()
            self.backbone.switch_out_indices()
        else:
            self.backbone.switch_strides(self.test_cfg.strides)
            self.backbone.switch_out_indices(self.test_cfg.out_indices)
