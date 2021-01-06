import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import PLUGIN_LAYERS, ConvModule


@PLUGIN_LAYERS.register_module()
class SelfAttention(nn.Module):

    def __init__(self,
                 in_channels=512,
                 kernel_size=1,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 num_convs=0,
                 reduction=2,
                 use_residual=True,
                 normalize=False):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_convs = num_convs
        self.reduction = reduction
        self.use_residual = use_residual
        self.normalize = normalize
        mid_channels = in_channels // reduction
        out_channels = in_channels

        last_channels = in_channels
        convs = []
        for i in range(num_convs):
            is_last = i == num_convs - 1
            out_channels = out_channels if is_last else \
                mid_channels
            convs.append(
                ConvModule(
                    last_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    # no bn/relu on output
                    norm_cfg=self.norm_cfg if not is_last else None,
                    act_cfg=self.act_cfg if not is_last else None))
            last_channels = out_channels
        if len(convs) > 0:
            self.convs = nn.Sequential(*convs)
        else:
            self.convs = nn.Identity()

    def forward(self, query, key, value):
        assert query.shape == key.shape == value.shape
        print(query.shape)
        identity = query
        query = query.flatten(2)
        key = key.flatten(2)
        value = self.convs(value)
        value = value.flatten(2)
        if self.normalize:
            query = F.normalize(query, p=2, dim=1)
            key = F.normalize(key, p=2, dim=1)
        # [B, HxW, HxW]
        affinity = torch.einsum('bci, bcj->bij', key, query).contiguous()
        affinity = affinity.softmax(dim=1)
        out = torch.matmul(value, affinity).contiguous()
        out = out.view_as(identity)

        if self.use_residual:
            return out + identity
        else:
            return out
