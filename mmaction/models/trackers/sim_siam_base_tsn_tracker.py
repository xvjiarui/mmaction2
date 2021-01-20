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
class SimSiamBaseTSNTracker(VanillaTracker):
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
            self.att_to_target = self.train_cfg.get('att_to_target', True)
            self.feat_rescale = self.train_cfg.get('feat_rescale', False)
            self.pred_clip_index = self.train_cfg.get('pred_clip_index', 0)
            self.pred_frame_index = self.train_cfg.get('pred_frame_index', 0)
            self.target_clip_index = self.train_cfg.get(
                'target_clip_index', -1)
            self.target_frame_index = self.train_cfg.get(
                'target_frame_index', -1)
            self.aux_as_value = self.train_cfg.get('aux_as_value', True)
            self.transpose_temporal = self.train_cfg.get(
                'transpose_temporal', False)
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

    def forward_backbone(self, imgs1, imgs2, imgs_aux):
        assert isinstance(self.backbone, ResNet)
        if self.att_to_target:
            x1 = self.backbone(imgs1)
            with torch.no_grad():
                x_aux = self.backbone.conv1(imgs_aux)
                x_aux = self.backbone.maxpool(x_aux)
                att_feat = {}
                for i, layer_name in enumerate(self.backbone.res_layers):
                    res_layer = getattr(self.backbone, layer_name)
                    x_aux = res_layer(x_aux)
                    if i in self.att_indices and self.with_att_plugin:
                        att_feat[i] = x_aux

                x2 = self.backbone.conv1(imgs2)
                x2 = self.backbone.maxpool(x2)
                for i, layer_name in enumerate(self.backbone.res_layers):
                    res_layer = getattr(self.backbone, layer_name)
                    x2 = res_layer(x2)
                    if i in self.att_indices and self.with_att_plugin:
                        if self.aux_as_value:
                            value = att_feat[i]
                        else:
                            value = x2
                        x2 = self.att_plugin(x2, att_feat[i], value)
                        if self.feat_rescale:
                            x2 = x2 * 0.5
        else:
            att_feat = {}
            with torch.set_grad_enabled(self.bp_aux):
                x2 = self.backbone(imgs2)
                x_aux = self.backbone.conv1(imgs_aux)
                x_aux = self.backbone.maxpool(x_aux)
                for i, layer_name in enumerate(self.backbone.res_layers):
                    res_layer = getattr(self.backbone, layer_name)
                    x_aux = res_layer(x_aux)
                    if i in self.att_indices and self.with_att_plugin:
                        att_feat[i] = x_aux
            x1 = self.backbone.conv1(imgs1)
            x1 = self.backbone.maxpool(x1)
            for i, layer_name in enumerate(self.backbone.res_layers):
                res_layer = getattr(self.backbone, layer_name)
                x1 = res_layer(x1)
                if i in self.att_indices and self.with_att_plugin:
                    if self.aux_as_value:
                        value = att_feat[i]
                    else:
                        value = x1
                    x1 = self.att_plugin(x1, att_feat[i], value)
                    if self.feat_rescale:
                        x1 = x1 * 0.5

        return x1, x2

    def forward_train(self, imgs, label=None):
        # [B, N, C, T, H, W]
        assert imgs.ndim == 6
        clip_len = imgs.size(3)
        num_clips = imgs.size(1)
        if self.transpose_temporal:
            assert not self.intra_video
            if num_clips > 1:
                assert num_clips == clip_len == 2
                imgs1 = imgs[:, self.pred_clip_index, :, self.pred_frame_index]
                imgs2 = imgs[:, self.target_clip_index, :,
                             self.target_frame_index]
                imgs_aux = [
                    imgs[:, self.pred_clip_index, :,
                         (self.pred_frame_index + 1) % clip_len],
                    imgs[:, self.target_clip_index, :,
                         (self.target_frame_index + 1) % clip_len]
                ]
            else:
                imgs1 = imgs[:, 0, :, self.pred_frame_index]
                imgs2 = imgs[:, 0, :, self.target_frame_index]
                imgs_aux = [
                    imgs[:, 0, :,
                         (self.pred_frame_index + 1) % (clip_len // 2)],
                    imgs[:, 0, :, clip_len // 2 +
                         (self.target_frame_index + 1) % (clip_len // 2)]
                ]

        else:
            assert num_clips >= 2
            imgs = [
                video2images(imgs[:,
                                  i].contiguous().reshape(-1, *imgs.shape[2:]))
                for i in range(num_clips)
            ]
            imgs1 = imgs[self.pred_clip_index]
            imgs2 = imgs[self.target_clip_index]
            imgs_aux = imgs[self.pred_clip_index + 1:self.target_clip_index]
        assert len(imgs_aux) >= 1
        if self.att_to_target:
            imgs_aux.reverse()
        x11, x12 = self.forward_backbone(imgs1, imgs2, imgs_aux[0])
        x22, x21 = self.forward_backbone(imgs2, imgs1, imgs_aux[-1])
        losses = dict()
        if self.with_img_head:
            loss_img_head = self.forward_img_head(x11, x12, x22, x21, clip_len)
            losses.update(add_prefix(loss_img_head, prefix='img_head'))

        return losses
