from mmaction.utils import add_prefix
from .. import builder
from ..common import images2video, video2images
from ..registry import TRACKERS
from .vanilla_tracker import VanillaTracker


@TRACKERS.register_module()
class SimSiamPairTracker(VanillaTracker):
    """3D recognizer model framework."""

    def __init__(self, *args, backbone, img_head=None, **kwargs):
        super().__init__(*args, backbone=backbone, **kwargs)
        if img_head is not None:
            self.img_head = builder.build_head(img_head)
        self.init_extra_weights()
        if self.train_cfg is not None:
            self.multi_pair = self.train_cfg.get('multi_pair', False)
            self.pair_scale = self.train_cfg.get('pair_scale', False)

    @property
    def with_img_head(self):
        """bool: whether the detector has img head"""
        return hasattr(self, 'img_head') and self.img_head is not None

    def init_extra_weights(self):
        if self.with_img_head:
            self.img_head.init_weights()

    def forward_img_head(self, x1, x2, clip_len):
        losses = dict()
        z1, p1 = self.img_head(x1)
        z2, p2 = self.img_head(x2)
        loss_weight = 1. / clip_len if self.pair_scale else 1.
        losses.update(
            add_prefix(
                self.img_head.loss(p1, z1, p2, z2, weight=loss_weight),
                prefix='0'))
        if self.multi_pair:
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

    def forward_train(self, imgs, grids=None, label=None):
        # [B, N, C, T, H, W]
        assert imgs.size(1) == 2
        assert imgs.ndim == 6
        clip_len = imgs.size(3)
        imgs1 = video2images(imgs[:,
                                  0].contiguous().reshape(-1, *imgs.shape[2:]))
        imgs2 = video2images(imgs[:,
                                  1].contiguous().reshape(-1, *imgs.shape[2:]))
        x1 = self.backbone(imgs1)
        x2 = self.backbone(imgs2)
        losses = dict()
        if self.with_img_head:
            loss_img_head = self.forward_img_head(x1, x2, clip_len)
            losses.update(add_prefix(loss_img_head, prefix='img_head'))

        return losses
