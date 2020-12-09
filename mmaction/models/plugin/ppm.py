import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import PLUGIN_LAYERS, ConvModule


@PLUGIN_LAYERS.register_module()
class PixelPro(nn.Module):

    def __init__(self,
                 in_channels,
                 kernel_size=1,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 num_convs=1,
                 reduction=2,
                 gamma=2.,
                 use_residual=False,
                 normalize=True):
        super(PixelPro, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_convs = num_convs
        self.reduction = reduction
        self.gamma = gamma
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

    def forward(self, x):
        identity = x
        g = self.convs(x)
        g = g.flatten(2)
        x = x.flatten(2)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        # [B, HxW, HxW]
        affinity = torch.einsum('bci, bcj->bij', x, x).contiguous()
        affinity = affinity.clamp(min=0)**self.gamma

        out = torch.matmul(g, affinity).contiguous()
        out = out.view_as(identity)
        if self.use_residual:
            return out + identity
        else:
            return out
