import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops.point_sample import generate_grid

from ..builder import build_loss
from ..common import center2bbox, compute_affinity
from ..registry import HEADS


@HEADS.register_module()
class UVCHead(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_feat (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 num_convs=0,
                 kernel_size=1,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=None,
                 act_cfg=None,
                 with_norm=True,
                 loss_feat=dict(type='CosineSimLoss'),
                 loss_aff=dict(type='ConcentrateLoss'),
                 loss_bbox=dict(type='MSELoss'),
                 dropout_ratio=0,
                 init_std=0.01,
                 temperature=1.,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_convs = num_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_norm = with_norm
        self.loss_feat = build_loss(loss_feat)
        self.loss_aff = build_loss(loss_aff)
        self.loss_bbox = build_loss(loss_bbox)
        if num_convs > 0:
            convs = []
            convs.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            for i in range(num_convs - 2):
                convs.append(
                    ConvModule(
                        self.channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            if num_convs > 1:
                convs.append(
                    ConvModule(
                        self.channels,
                        self.channels,
                        kernel_size=1,
                        padding=0,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=None,
                        act_cfg=None))
            self.convs = nn.Sequential(*convs)
        else:
            self.convs = None

        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.temperature = temperature
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def get_tar_bboxes(self, ref_crop_x, tar_x):
        ref_crop_x = self(ref_crop_x)
        tar_x = self(tar_x)
        # [N, tar_w*tar_h, 2]
        tar_grid = generate_grid(
            tar_x.size(0), tar_x.shape[2:], device=ref_crop_x.device)
        # TODO check
        # [N, tar_w*tar_h, 2]
        tar_coords = torch.stack([
            tar_grid[..., 0] * tar_x.size(3), tar_grid[..., 1] * tar_x.size(2)
        ],
                                 dim=2).contiguous()
        # [N, ref_w*ref_h, tar_w*tar_h]
        aff_ref_tar = compute_affinity(
            ref_crop_x,
            tar_x,
            temperature=self.temperature,
            normalize=self.with_norm,
            softmax_dim=2).contiguous()
        # [N, ref_w*ref_h, 2]
        ref_coords = torch.bmm(aff_ref_tar, tar_coords)
        # [N, 2]
        ref_center = torch.mean(ref_coords, dim=1)
        tar_bboxes = center2bbox(ref_center, ref_crop_x.shape[2:],
                                 tar_x.shape[2:])

        return tar_bboxes

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.convs is not None:
            x = self.convs(x)
        return x

    def loss(self, src_x, dst_x, weight=1.):

        losses = dict()
        losses['loss_feat'] = self.loss_feat(src_x, dst_x) * weight
        losses['loss_aff'] = self.loss_aff(src_x, dst_x) * weight

        return losses
