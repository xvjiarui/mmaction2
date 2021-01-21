from collections import defaultdict

import torch
from mmcv.cnn import build_plugin_layer

from mmaction.utils import add_prefix
from .. import builder
from ..backbones import ResNet
from ..heads import SimSiamHead
from ..registry import TRACKERS
from .vanilla_tracker import VanillaTracker


@TRACKERS.register_module()
class SimSiamBaseSTSNTracker(VanillaTracker):
    """3D recognizer model framework."""

    def __init__(self, *args, backbone, att_plugin, img_head=None, **kwargs):
        super().__init__(*args, backbone=backbone, **kwargs)
        if img_head is not None:
            self.img_head = builder.build_head(img_head)
        if att_plugin is not None:
            self.att_plugin = build_plugin_layer(att_plugin)[1]
        self.init_extra_weights()
        if self.train_cfg is not None:
            self.att_indices = self.train_cfg.get('att_indices')
            self.pred_clip_index = self.train_cfg.get('pred_clip_index', 0)
            self.pred_frame_index = self.train_cfg.get('pred_frame_index', 0)
            self.target_clip_index = self.train_cfg.get(
                'target_clip_index', -1)
            self.target_frame_index = self.train_cfg.get(
                'target_frame_index', -1)
            self.self_as_value = self.train_cfg.get('self_as_value', True)
            self.bp_aux = self.train_cfg.get('bp_aux', False)

    @property
    def with_img_head(self):
        """bool: whether the detector has img head"""
        return hasattr(self, 'img_head') and self.img_head is not None

    @property
    def with_att_plugin(self):
        """bool: whether the detector has img head"""
        return hasattr(self, 'att_plugin') and self.att_plugin is not None

    def init_extra_weights(self):
        if self.with_img_head:
            self.img_head.init_weights()

    def forward_img_head(self, x11, x12, x22, x21, clip_len):
        assert isinstance(self.img_head, SimSiamHead)
        losses = dict()
        _, p1 = self.img_head(x11)
        _, p2 = self.img_head(x22)
        with torch.no_grad():
            z1 = self.img_head.forward_projection(x21)
            z2 = self.img_head.forward_projection(x12)
        losses.update(
            add_prefix(
                self.img_head.loss(p1, z1, p2, z2, weight=1.), prefix='0'))
        return losses

    def forward_backbone(self, imgs1, imgs2, imgs_aux_list):
        assert isinstance(self.backbone, ResNet)
        att_feat = defaultdict(list)
        with torch.no_grad():
            x2 = self.backbone(imgs2)
        with torch.set_grad_enabled(self.bp_aux):
            for imgs_aux in imgs_aux_list:
                x_aux = self.backbone.conv1(imgs_aux)
                x_aux = self.backbone.maxpool(x_aux)
                for i, layer_name in enumerate(self.backbone.res_layers):
                    res_layer = getattr(self.backbone, layer_name)
                    x_aux = res_layer(x_aux)
                    if i in self.att_indices and self.with_att_plugin:
                        att_feat[i].append(x_aux)
        x1 = self.backbone.conv1(imgs1)
        x1 = self.backbone.maxpool(x1)
        for i, layer_name in enumerate(self.backbone.res_layers):
            res_layer = getattr(self.backbone, layer_name)
            x1 = res_layer(x1)
            if i in self.att_indices and self.with_att_plugin:
                if self.self_as_value:
                    att_feat[i].append(x1)
                concat_att = torch.stack(att_feat[i], dim=2)
                x1 = self.att_plugin(x1, concat_att, concat_att)

        return x1, x2

    def forward_train(self, imgs, label=None):
        # [B, N, C, T, H, W]
        assert imgs.ndim == 6
        clip_len = imgs.size(3)
        num_clips = imgs.size(1)
        self.pred_clip_index = self.pred_clip_index % num_clips
        self.pred_frame_index = self.pred_frame_index % clip_len
        self.target_clip_index = self.target_clip_index % num_clips
        self.target_frame_index = self.target_frame_index % clip_len
        if num_clips > 1:
            assert num_clips == 2
            imgs1 = imgs[:, self.pred_clip_index, :, self.pred_frame_index]
            imgs2 = imgs[:, self.target_clip_index, :, self.target_frame_index]
            imgs_aux = []
            for i in range(clip_len):
                if i != self.pred_frame_index:
                    imgs_aux.append(imgs[:, self.pred_clip_index, :, i])
            for i in range(clip_len):
                if i != self.target_frame_index:
                    imgs_aux.append(imgs[:, self.target_clip_index, :, i])
        else:
            assert clip_len % 2 == 0
            imgs1 = imgs[:, 0, :, self.pred_frame_index]
            imgs2 = imgs[:, 0, :, self.target_frame_index]
            imgs_aux = []
            for i in range(clip_len // 2):
                if i != self.pred_frame_index:
                    imgs_aux.append(imgs[:, 0, :, i])
            for i in range(clip_len // 2, clip_len):
                if i != self.target_frame_index:
                    imgs_aux.append(imgs[:, 0, :, i])

        assert len(imgs_aux) >= 1
        x11, x12 = self.forward_backbone(imgs1, imgs2,
                                         imgs_aux[:len(imgs_aux) // 2])
        x22, x21 = self.forward_backbone(imgs2, imgs1,
                                         imgs_aux[len(imgs_aux) // 2:])
        losses = dict()
        if self.with_img_head:
            loss_img_head = self.forward_img_head(x11, x12, x22, x21, clip_len)
            losses.update(add_prefix(loss_img_head, prefix='img_head'))

        return losses
