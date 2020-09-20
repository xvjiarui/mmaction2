import kornia.augmentation as K
import torch.nn as nn

from ..registry import WALKERS
from .vanilla_tracker import VanillaTracker


@WALKERS.register_module()
class SpaceTimeWalker(VanillaTracker):
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
