import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmaction.utils import add_suffix
from ..registry import HEADS
from .base import BaseHead


@HEADS.register_module()
class WalkerHead(BaseHead):
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
                 channels,
                 num_convs=1,
                 kernel_size=1,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=None,
                 act_cfg=None,
                 with_norm=True,
                 loss_cls=dict(type='NLLLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 temperature=0.07,
                 walk_len=7,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.channels = channels
        self.num_convs = num_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_norm = with_norm
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

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.temperature = temperature
        assert walk_len >= 1
        self.walk_len = walk_len
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def walk(self, x, batches, clip_len):
        channels = x.size(1)
        # [N, T, P, C]
        x = x.reshape(batches, clip_len, -1, channels)
        num_patches = x.size(2)

        # [N, T-1, P, P]
        # TODO check debug
        affinity_forward = torch.einsum('btpc,btqc->btpq', x[:, :-1],
                                        x[:, 1:]) / self.temperature
        # affinity_forward_ = torch.matmul(x[:, :-1].reshape(
        #     batches * (clip_len-1), num_patches, channels), x[:, 1:].reshape(
        #     batches * (clip_len-1), num_patches, channels).transpose(
        #     -1, -2)).reshape(batches, clip_len-1, num_patches,
        #                      num_patches)/self.temperature
        # assert torch.allclose(affinity_forward, affinity_forward_)
        # [N, T-1, P, P]
        affinity_backward = affinity_forward.transpose(-1, -2)

        preds_list = []
        for step in range(1, min(clip_len, self.walk_len + 1)):
            # [N, P, P]
            preds = torch.eye(num_patches).to(x).unsqueeze(0).expand(
                batches, -1, -1)
            for t in range(step):
                # preds = torch.bmm(affinity_forward[:, t].softmax(dim=-1),
                #                   preds)
                preds = torch.bmm(preds, affinity_forward[:,
                                                          t].softmax(dim=-1))
            for t in reversed(range(step)):
                # preds = torch.bmm(affinity_backward[:, t].softmax(dim=-1),
                #                   preds)
                preds = torch.bmm(preds, affinity_backward[:,
                                                           t].softmax(dim=-1))
            # swap softmax dim to 1
            preds = preds.transpose(1, 2).log()
            preds_list.append(preds)

        return preds_list

    def get_targets(self, preds_list):
        labels_list = []
        for i in range(len(preds_list)):
            preds = preds_list[i]
            batches, num_patches = preds.shape[:2]
            # [N, P]
            labels = torch.arange(
                num_patches, dtype=torch.long,
                device=preds.device).unsqueeze(0).expand(batches, -1)

            labels_list.append(labels)

        return labels_list

    def forward(self, x, batches, clip_len):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        x = self.avg_pool(x)
        x = self.convs(x)
        if self.with_norm:
            x = F.normalize(x, p=2, dim=1)
        preds_list = self.walk(x, batches, clip_len)

        return preds_list

    def loss(self, preds_list):
        labels_list = self.get_targets(preds_list)
        losses = dict()
        for idx, (preds, labels) in enumerate(zip(preds_list, labels_list)):
            preds = preds.reshape(-1, preds.size(-1))
            labels = labels.reshape(-1)
            losses.update(
                add_suffix(super().loss(preds, labels), suffix=str(idx)))

        return losses
