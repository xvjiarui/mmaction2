import torch
from mmcv.cnn import build_plugin_layer

from mmaction.utils import add_prefix
from .. import builder
from ..backbones import ResNet
from ..common import images2video, video2images
from ..heads import SimSiamHead
from ..registry import TRACKERS
from .vanilla_tracker import VanillaTracker


@TRACKERS.register_module()
class SimSiamBaseSingleTracker(VanillaTracker):
    """3D recognizer model framework."""

    def __init__(self, *args, backbone, att_plugin, img_head=None, **kwargs):
        super().__init__(*args, backbone=backbone, **kwargs)
        if img_head is not None:
            self.img_head = builder.build_head(img_head)
        if att_plugin is not None:
            self.att_plugin = build_plugin_layer(att_plugin)[1]
        self.init_extra_weights()
        if self.train_cfg is not None:
            self.intra_video = self.train_cfg.get('intra_video', False)
            self.att_indices = self.train_cfg.get('att_indices')

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
        loss_weight = 1. / clip_len if self.intra_video else 1.
        losses.update(
            add_prefix(
                self.img_head.loss(p1, z1, p2, z2, weight=loss_weight),
                prefix='0'))
        if self.intra_video:
            z2_v, p2_v = images2video(z2, clip_len), images2video(p2, clip_len)
            for i in range(1, clip_len):
                losses.update(
                    add_prefix(
                        self.img_head.loss(
                            p1,
                            z1,
                            video2images(p2_v.roll(i, dims=2)),
                            video2images(z2_v.roll(i, dims=2)),
                            weight=loss_weight),
                        prefix=f'{i}'))
        return losses

    def forward_backbone(self, imgs1, imgs2):
        assert isinstance(self.backbone, ResNet)
        x1 = self.backbone.conv1(imgs1)
        x1 = self.backbone.maxpool(x1)
        att_feat = {}
        for i, layer_name in enumerate(self.backbone.res_layers):
            res_layer = getattr(self.backbone, layer_name)
            x1 = res_layer(x1)
            if i in self.att_indices and self.with_att_plugin:
                att_feat[i] = x1

        with torch.no_grad():
            x2 = self.backbone.conv1(imgs2)
            x2 = self.backbone.maxpool(x2)
            for i, layer_name in enumerate(self.backbone.res_layers):
                res_layer = getattr(self.backbone, layer_name)
                x2 = res_layer(x2)
                if i in self.att_indices and self.with_att_plugin:
                    x2 = self.att_plugin(x2, att_feat[i], x2)

        return x1, x2

    def forward_train(self, imgs, label=None):
        # [B, N, C, T, H, W]
        assert imgs.size(1) == 2
        assert imgs.ndim == 6
        clip_len = imgs.size(3)
        imgs1 = video2images(imgs[:,
                                  0].contiguous().reshape(-1, *imgs.shape[2:]))
        imgs2 = video2images(imgs[:,
                                  1].contiguous().reshape(-1, *imgs.shape[2:]))
        x11, x12 = self.forward_backbone(imgs1, imgs2)
        x22, x21 = self.forward_backbone(imgs2, imgs1)
        losses = dict()
        if self.with_img_head:
            loss_img_head = self.forward_img_head(x11, x12, x22, x21, clip_len)
            losses.update(add_prefix(loss_img_head, prefix='img_head'))

        return losses
