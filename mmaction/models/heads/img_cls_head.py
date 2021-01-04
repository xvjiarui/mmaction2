import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .base import BaseHead


@HEADS.register_module()
class ImageClsHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 norm_cfg=dict(type='BN'),
                 num_projection_fcs=3,
                 projection_mid_channels=2048,
                 projection_out_channels=2048,
                 drop_projection_fc=False,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        projection_fcs = []
        if norm_cfg['type'] == 'SyncBN':
            BatchNorm1d = nn.SyncBatchNorm
        else:
            BatchNorm1d = nn.BatchNorm1d
        last_channels = in_channels
        for i in range(num_projection_fcs):
            is_last = i == num_projection_fcs - 1
            out_channels = projection_out_channels if is_last else \
                projection_mid_channels
            projection_fcs.append(nn.Linear(last_channels, out_channels))
            projection_fcs.append(BatchNorm1d(out_channels))
            # no relu on output
            if not is_last:
                projection_fcs.append(nn.ReLU())
                if drop_projection_fc:
                    projection_fcs.append(nn.Dropout(p=self.dropout_ratio))
            last_channels = out_channels
        if len(projection_fcs):
            self.projection_fcs = nn.Sequential(*projection_fcs)
        else:
            self.projection_fcs = nn.Identity()
        self.fc_cls = nn.Linear(last_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = nn.Identity()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.flatten(1)
        x = self.projection_fcs(x)
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
