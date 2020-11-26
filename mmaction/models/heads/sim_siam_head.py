import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import build_loss
from ..registry import HEADS


@HEADS.register_module()
class SimSiamHead(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_feat (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 in_channels,
                 conv_mid_channels=2048,
                 conv_out_channles=2048,
                 num_convs=0,
                 kernel_size=1,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=None,
                 num_projection_fcs=3,
                 projection_mid_channels=2048,
                 projection_out_channels=2048,
                 num_predictor_fcs=2,
                 predictor_mid_channels=512,
                 predictor_out_channels=2048,
                 with_norm=True,
                 loss_feat=dict(type='CosineSimLoss', negative=False),
                 spatial_type='avg',
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_convs = num_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_norm = with_norm
        self.loss_feat = build_loss(loss_feat)
        convs = []
        last_channels = in_channels
        for i in range(num_convs):
            is_last = i == num_convs - 1
            out_channels = conv_out_channles if is_last else conv_mid_channels
            convs.append(
                ConvModule(
                    last_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg if not is_last else None,
                    act_cfg=self.act_cfg if not is_last else None))
            last_channels = out_channels
        if len(convs) > 0:
            self.convs = nn.Sequential(*convs)
        else:
            self.convs = nn.Identity()

        if norm_cfg['type'] == 'SyncBN':
            BatchNorm1d = nn.SyncBatchNorm
        else:
            BatchNorm1d = nn.BatchNorm1d

        projection_fcs = []
        for i in range(num_projection_fcs):
            is_last = i == num_projection_fcs - 1
            out_channels = projection_out_channels if is_last else \
                projection_mid_channels
            projection_fcs.append(nn.Linear(last_channels, out_channels))
            projection_fcs.append(BatchNorm1d(out_channels))
            # no relu on output
            if not is_last:
                projection_fcs.append(nn.ReLU())
            last_channels = out_channels
        if len(projection_fcs):
            self.projection_fcs = nn.Sequential(*projection_fcs)
        else:
            self.projection_fcs = nn.Identity()

        predictor_fcs = []
        for i in range(num_predictor_fcs):
            is_last = i == num_predictor_fcs - 1
            out_channels = predictor_out_channels if is_last else \
                predictor_mid_channels
            predictor_fcs.append(nn.Linear(last_channels, out_channels))
            if not is_last:
                predictor_fcs.append(BatchNorm1d(out_channels))
                predictor_fcs.append(nn.ReLU())
            last_channels = out_channels
        if len(predictor_fcs):
            self.predictor_fcs = nn.Sequential(*predictor_fcs)
        else:
            self.predictor_fcs = nn.Identity()

        assert spatial_type in ['avg', None]
        self.spatial_type = spatial_type
        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = nn.Identity()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        x = self.convs(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        z = self.projection_fcs(x)
        p = self.predictor_fcs(z)

        return z, p

    def loss(self, p1, z1, p2, z2, weight=1.):

        losses = dict()

        loss_feat = self.loss_feat(p1, z2.detach()) * 0.5 + self.loss_feat(
            p2, z1.detach()) * 0.5
        losses['loss_feat'] = loss_feat * weight
        return losses
