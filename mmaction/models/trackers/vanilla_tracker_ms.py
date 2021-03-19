import os.path as osp
import tempfile

import mmcv
import numpy as np
import torch
import torch.nn.functional as F

from ..backbones import ResNet
from ..common import (StrideContext, cat, images2video, interpolate3d,
                      masked_attention_efficient, mean_list, normalize_logit,
                      pil_nearest_interpolate, spatial_neighbor, video2images)
from ..registry import TRACKERS
from .base import BaseTracker


@TRACKERS.register_module()
class VanillaTracker(BaseTracker):
    """Pixel Tracker framework."""

    def __init__(self, *args, **kwargs):
        super(VanillaTracker, self).__init__(*args, **kwargs)
        self.save_np = self.test_cfg.get('save_np', False)

    @property
    def stride(self):
        if self.with_neck:
            end_index = self.backbone.original_out_indices[self.neck.out_index]
        else:
            end_index = self.backbone.original_out_indices[0]
        return np.prod(self.backbone.strides[:end_index + 1]) * 4

    def extract_feat_test(self, imgs):
        outs = []
        if self.test_cfg.get('all_blocks', False):
            assert isinstance(self.backbone, ResNet)
            x = self.backbone.conv1(imgs)
            x = self.backbone.maxpool(x)
            outs = []
            for i, layer_name in enumerate(self.backbone.res_layers):
                res_layer = getattr(self.backbone, layer_name)
                if i in self.test_cfg.out_indices:
                    for block in res_layer:
                        x = block(x)
                        outs.append(x)
                else:
                    x = res_layer(x)
            return tuple(outs)
        if self.with_neck:
            if self.test_cfg.get('use_fpn', True):
                neck_out = self.extract_feat(imgs)
                if isinstance(neck_out, (tuple, list)):
                    neck_out_indices = self.test_cfg.get(
                        'neck_out_indices', [0])
                    for ni in neck_out_indices:
                        outs.append(neck_out[ni])
                else:
                    outs.append(neck_out)
            # switch manually
            if self.test_cfg.get('use_backbone', False):
                with StrideContext(self.backbone, self.test_cfg.strides,
                                   self.test_cfg.out_indices):
                    backbone_out = self.backbone(imgs)
                    if isinstance(backbone_out, (tuple, list)):
                        outs.extend(list(backbone_out))
                    else:
                        outs.append(backbone_out)
            if len(outs) == 1:
                return outs[0]
            return tuple(outs)
        else:
            if self.test_cfg.get('concat_feats', False):
                feats = self.extract_feat(imgs)
                feats = feats + (cat(feats, dim=1), )
                return feats
            return self.extract_feat(imgs)

    def extract_single_feat(self, imgs, idx):
        feats = self.extract_feat_test(imgs)
        if isinstance(feats, (tuple, list)):
            return feats[idx]
        else:
            return feats

    def get_feats(self, imgs, num_feats):
        assert imgs.shape[0] == 1
        batch_step = self.test_cfg.get('batch_step', 10)
        feat_bank = [[] for _ in range(num_feats)]
        clip_len = imgs.size(2)
        imgs = video2images(imgs)
        for batch_ptr in range(0, clip_len, batch_step):
            feats = self.extract_feat_test(imgs[batch_ptr:batch_ptr +
                                                batch_step])
            if isinstance(feats, tuple):
                assert len(feats) == len(feat_bank)
                for i in range(len(feats)):
                    feat_bank[i].append(feats[i].cpu())
            else:
                feat_bank[0].append(feats.cpu())
        for i in range(num_feats):
            feat_bank[i] = images2video(
                torch.cat(feat_bank[i], dim=0), clip_len)
            assert feat_bank[i].size(2) == clip_len

        return feat_bank

    def forward_train(self, imgs, labels=None):
        raise NotImplementedError

    def simple_test(self, imgs, ref_seg_map, img_shape):
        """Defines the computation performed at every call when evaluation and
        testing."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        clip_len = imgs.size(2)
        if imgs.shape[2:] != img_shape:
            imgs = images2video(
                F.interpolate(
                    video2images(imgs),
                    size=img_shape,
                    mode='bilinear',
                    align_corners=False), clip_len)
        # get target shape
        dummy_feat = self.extract_feat_test(imgs[0:1, :, 0])
        if isinstance(dummy_feat, (list, tuple)):
            feat_shapes = [_.shape for _ in dummy_feat]
        else:
            feat_shapes = [dummy_feat.shape]
        feat_bank = self.get_feats(imgs, len(dummy_feat))
        seg_logits_all = []
        for feat_idx, feat_shape in enumerate(feat_shapes):
            input_onehot = ref_seg_map.ndim == 4
            if not input_onehot:
                resized_seg_map = pil_nearest_interpolate(
                    ref_seg_map.unsqueeze(1),
                    size=feat_shape[2:]).squeeze(1).long()
                resized_seg_map = F.one_hot(resized_seg_map).permute(
                    0, 3, 1, 2).float()
                ref_seg_map = F.interpolate(
                    ref_seg_map.unsqueeze(1), size=img_shape,
                    mode='nearest').squeeze(1)
            else:
                resized_seg_map = F.interpolate(
                    ref_seg_map,
                    size=feat_shape[2:],
                    mode='bilinear',
                    align_corners=False).float()
                ref_seg_map = F.interpolate(
                    ref_seg_map,
                    size=img_shape,
                    mode='bilinear',
                    align_corners=False)
            seg_bank = []

            neighbor_range = self.test_cfg.get('neighbor_range', None)
            if neighbor_range is not None:
                spatial_neighbor_mask = spatial_neighbor(
                    feat_shape[0],
                    *feat_shape[2:],
                    neighbor_range=neighbor_range,
                    device=imgs.device,
                    dtype=imgs.dtype,
                    mode='circle')
            else:
                spatial_neighbor_mask = None

            seg_bank.append(resized_seg_map.cpu())
            for frame_idx in range(1, clip_len):
                key_start = max(0, frame_idx - self.test_cfg.precede_frames)
                query_feat = feat_bank[feat_idx][:, :,
                                                 frame_idx].to(imgs.device)
                key_feat = feat_bank[feat_idx][:, :, key_start:frame_idx].to(
                    imgs.device)
                value_logits = torch.stack(
                    seg_bank[key_start:frame_idx], dim=2).to(imgs.device)
                if self.test_cfg.get('with_first', True):
                    key_feat = torch.cat([
                        feat_bank[feat_idx][:, :, 0:1].to(imgs.device),
                        key_feat
                    ],
                                         dim=2)
                    value_logits = cat([
                        seg_bank[0].unsqueeze(2).to(imgs.device), value_logits
                    ],
                                       dim=2)
                seg_logit = masked_attention_efficient(
                    query_feat,
                    key_feat,
                    value_logits,
                    spatial_neighbor_mask,
                    temperature=self.test_cfg.temperature,
                    topk=self.test_cfg.topk,
                    normalize=self.test_cfg.get('with_norm', True),
                    non_mask_len=0 if self.test_cfg.get(
                        'with_first_neighbor', True) else 1)
                seg_bank.append(seg_logit.cpu())
            seg_logits = torch.stack(
                [normalize_logit(seg) for seg in seg_bank], dim=2)
            seg_logits_all.append(seg_logits)
        if self.test_cfg.get('aug_logits', False):
            seg_logits_all.append(mean_list(seg_logits_all))
        return seg_logits_all

    def aug_test(self, imgs, ref_seg_map, img_meta):
        all_seg_preds = []
        if self.test_cfg.get('ms_test', False):
            test_scales = (0.5, 0.75, 1., 1.25, 1.5)
            test_shapes = []
            input_shape = imgs.shape[-2:]
            for s in test_scales:
                test_shapes.append(
                    (int(input_shape[0] * s), int(input_shape[1] * s)))
        else:
            test_shapes = [imgs.shape[-2:]]
        aug_seg_preds = None
        for test_shape in test_shapes:
            if aug_seg_preds is None:
                aug_seg_preds = self.simple_test(imgs, ref_seg_map, test_shape)
                aug_seg_preds = [
                    interpolate3d(
                        aug_seg_pred,
                        size=img_meta[0]['original_shape'][:2],
                        mode='bilinear',
                        align_corners=False) for aug_seg_pred in aug_seg_preds
                ]
            else:
                simple_seg_logits = self.simple_test(imgs, ref_seg_map,
                                                     test_shape)
                for i in range(len(aug_seg_preds)):
                    aug_seg_preds[i] += interpolate3d(
                        simple_seg_logits[i],
                        size=img_meta[0]['original_shape'][:2],
                        mode='bilinear',
                        align_corners=False)
        for i in range(len(aug_seg_preds)):
            aug_seg_preds[i] = aug_seg_preds[i].argmax(
                axis=1).byte().cpu().numpy()
            if self.save_np:
                assert aug_seg_preds[i].shape[0] == 1
                eval_dir = '.eval'
                mmcv.mkdir_or_exist(eval_dir)
                temp_file = tempfile.NamedTemporaryFile(
                    dir=eval_dir, suffix='.npy', delete=False)
                file_path = osp.join(eval_dir, temp_file.name)
                np.save(file_path, aug_seg_preds[i][0])
                all_seg_preds.append(file_path)
            else:
                all_seg_preds.append(aug_seg_preds[i])

        if self.save_np:
            if len(all_seg_preds) > 1:
                return [all_seg_preds]
            else:
                return [all_seg_preds[0]]
        else:
            if len(all_seg_preds) > 1:
                all_seg_preds = np.stack(all_seg_preds, axis=1)
            else:
                all_seg_preds = all_seg_preds[0]
            # unravel batch dim
            return list(all_seg_preds)

    def forward_test(self, imgs, ref_seg_map, img_meta):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self.aug_test(imgs, ref_seg_map, img_meta)
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        clip_len = imgs.size(2)
        # get target shape
        dummy_feat = self.extract_feat_test(imgs[0:1, :, 0])
        if isinstance(dummy_feat, (list, tuple)):
            feat_shapes = [_.shape for _ in dummy_feat]
        else:
            feat_shapes = [dummy_feat.shape]
        all_seg_preds = []
        feat_bank = self.get_feats(imgs, len(dummy_feat))
        seg_bank_all = []
        for feat_idx, feat_shape in enumerate(feat_shapes):
            input_onehot = ref_seg_map.ndim == 4
            if not input_onehot:
                resized_seg_map = pil_nearest_interpolate(
                    ref_seg_map.unsqueeze(1),
                    size=feat_shape[2:]).squeeze(1).long()
                resized_seg_map = F.one_hot(resized_seg_map).permute(
                    0, 3, 1, 2).float()
                ref_seg_map = F.interpolate(
                    ref_seg_map.unsqueeze(1),
                    size=img_meta[0]['original_shape'][:2],
                    mode='nearest').squeeze(1)
            else:
                resized_seg_map = F.interpolate(
                    ref_seg_map,
                    size=feat_shape[2:],
                    mode='bilinear',
                    align_corners=False).float()
                ref_seg_map = F.interpolate(
                    ref_seg_map,
                    size=img_meta[0]['original_shape'][:2],
                    mode='bilinear',
                    align_corners=False)
            seg_bank = []

            seg_preds = [ref_seg_map.detach().cpu().numpy()]
            neighbor_range = self.test_cfg.get('neighbor_range', None)
            if neighbor_range is not None:
                spatial_neighbor_mask = spatial_neighbor(
                    feat_shape[0],
                    *feat_shape[2:],
                    neighbor_range=neighbor_range,
                    device=imgs.device,
                    dtype=imgs.dtype,
                    mode='circle')
            else:
                spatial_neighbor_mask = None

            seg_bank.append(resized_seg_map.cpu())
            for frame_idx in range(1, clip_len):
                key_start = max(0, frame_idx - self.test_cfg.precede_frames)
                query_feat = feat_bank[feat_idx][:, :,
                                                 frame_idx].to(imgs.device)
                key_feat = feat_bank[feat_idx][:, :, key_start:frame_idx].to(
                    imgs.device)
                value_logits = torch.stack(
                    seg_bank[key_start:frame_idx], dim=2).to(imgs.device)
                if self.test_cfg.get('with_first', True):
                    key_feat = torch.cat([
                        feat_bank[feat_idx][:, :, 0:1].to(imgs.device),
                        key_feat
                    ],
                                         dim=2)
                    value_logits = cat([
                        seg_bank[0].unsqueeze(2).to(imgs.device), value_logits
                    ],
                                       dim=2)
                seg_logit = masked_attention_efficient(
                    query_feat,
                    key_feat,
                    value_logits,
                    spatial_neighbor_mask,
                    temperature=self.test_cfg.temperature,
                    topk=self.test_cfg.topk,
                    normalize=self.test_cfg.get('with_norm', True),
                    non_mask_len=0 if self.test_cfg.get(
                        'with_first_neighbor', True) else 1)
                seg_bank.append(seg_logit.cpu())

                seg_pred = F.interpolate(
                    seg_logit,
                    size=img_meta[0]['original_shape'][:2],
                    mode='bilinear',
                    align_corners=False)
                if not input_onehot:
                    seg_pred_min = seg_pred.view(*seg_pred.shape[:2], -1).min(
                        dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
                    seg_pred_max = seg_pred.view(*seg_pred.shape[:2], -1).max(
                        dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
                    normalized_seg_pred = (seg_pred - seg_pred_min) / (
                        seg_pred_max - seg_pred_min + 1e-12)
                    seg_pred = torch.where(seg_pred_max > 0,
                                           normalized_seg_pred, seg_pred)
                    seg_pred = seg_pred.argmax(dim=1)
                    seg_pred = F.interpolate(
                        seg_pred.byte().unsqueeze(1),
                        size=img_meta[0]['original_shape'][:2],
                        mode='nearest').squeeze(1)
                seg_preds.append(seg_pred.detach().cpu().numpy())

            seg_preds = np.stack(seg_preds, axis=1)
            if self.save_np:
                assert seg_preds.shape[0] == 1
                eval_dir = '.eval'
                mmcv.mkdir_or_exist(eval_dir)
                temp_file = tempfile.NamedTemporaryFile(
                    dir=eval_dir, suffix='.npy', delete=False)
                file_path = osp.join(eval_dir, temp_file.name)
                np.save(file_path, seg_preds[0])
                all_seg_preds.append(file_path)
            else:
                all_seg_preds.append(seg_preds)
            seg_bank_all.append(seg_bank)
        seg_preds_aug = []
        for frame_idx in range(clip_len):
            seg_pred_aug = None
            for feat_idx, feat_shape in enumerate(feat_shapes):
                seg_logit = seg_bank_all[feat_idx][frame_idx]
                seg_pred = F.interpolate(
                    seg_logit,
                    size=img_meta[0]['original_shape'][:2],
                    mode='bilinear',
                    align_corners=False)
                if not input_onehot:
                    seg_pred_min = seg_pred.view(*seg_pred.shape[:2], -1).min(
                        dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
                    seg_pred_max = seg_pred.view(*seg_pred.shape[:2], -1).max(
                        dim=-1)[0].view(*seg_pred.shape[:2], 1, 1)
                    normalized_seg_pred = (seg_pred - seg_pred_min) / (
                        seg_pred_max - seg_pred_min + 1e-12)
                    seg_pred = torch.where(seg_pred_max > 0,
                                           normalized_seg_pred, seg_pred)
                    if seg_pred_aug is None:
                        seg_pred_aug = seg_pred
                    else:
                        seg_pred_aug += seg_pred
            seg_pred_aug = seg_pred_aug.argmax(dim=1)
            seg_pred_aug = F.interpolate(
                seg_pred_aug.byte().unsqueeze(1),
                size=img_meta[0]['original_shape'][:2],
                mode='nearest').squeeze(1)
            seg_preds_aug.append(seg_pred_aug.detach().cpu().numpy())
        seg_preds_aug = np.stack(seg_preds_aug, axis=1)
        if self.save_np:
            assert seg_preds_aug.shape[0] == 1
            eval_dir = '.eval'
            mmcv.mkdir_or_exist(eval_dir)
            temp_file = tempfile.NamedTemporaryFile(
                dir=eval_dir, suffix='.npy', delete=False)
            file_path = osp.join(eval_dir, temp_file.name)
            np.save(file_path, seg_preds_aug[0])
            all_seg_preds.append(file_path)
        else:
            all_seg_preds.append(seg_preds_aug)

        if self.save_np:
            if len(all_seg_preds) > 1:
                return [all_seg_preds]
            else:
                return [all_seg_preds[0]]
        else:
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
            if hasattr(self.backbone, 'switch_strides'):
                self.backbone.switch_strides()
            if hasattr(self.backbone, 'switch_out_indices'):
                self.backbone.switch_out_indices()
        else:
            if not self.with_neck:
                if hasattr(self.backbone, 'switch_strides'):
                    self.backbone.switch_strides(self.test_cfg.strides)
                if hasattr(self.backbone, 'switch_out_indices'):
                    self.backbone.switch_out_indices(self.test_cfg.out_indices)
