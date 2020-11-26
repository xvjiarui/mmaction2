import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmaction.utils import add_suffix
from ..registry import HEADS
from .base import BaseHead


@HEADS.register_module()
class WalkerHeadV2(BaseHead):
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
                 dropout_ratio=0.1,
                 temperature=0.07,
                 walk_len=7,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        # assert loss_cls['type'] == 'NLLLoss'
        self.channels = channels
        self.num_convs = num_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_norm = with_norm
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
        self.temperature = temperature
        assert walk_len >= 1
        self.walk_len = walk_len

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def walk(self, x, batches, clip_len):
        channels = x.size(1)
        # [N, T, P, C]
        x = x.reshape(batches, clip_len, -1, channels)
        num_patches = x.size(2)

        # [N, T-1, P, P]
        affinity_forward = torch.einsum('btpc,btqc->btpq', x[:, :-1],
                                        x[:, 1:]) / self.temperature
        affinity_backward = affinity_forward.transpose(-1, -2)

        affinity_forward[
            torch.rand_like(affinity_forward) < self.dropout_ratio] = -1e20
        affinity_backward[
            torch.rand_like(affinity_backward) < self.dropout_ratio] = -1e20

        preds_list = []
        for step in range(1, min(clip_len, self.walk_len + 1)):
            # [N, P, P]
            preds = torch.eye(num_patches).to(x).unsqueeze(0).expand(
                batches, -1, -1)
            for t in range(step):
                preds = torch.bmm(affinity_forward[:, t].softmax(dim=-1),
                                  preds)
            for t in reversed(range(step)):
                preds = torch.bmm(affinity_backward[:, t].softmax(dim=-1),
                                  preds)
            preds_list.append(preds)

        return preds_list

    def forward(self, x, batches, clip_len):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        x = self.avg_pool(x)
        if self.convs is not None:
            x = self.convs(x)
        if self.with_norm:
            x = F.normalize(x, p=2, dim=1)
        preds_list = self.walk(x, batches, clip_len)

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

    def loss(self, preds_list):
        labels_list = self.get_targets(preds_list)
        losses = dict()
        for idx, (preds, labels) in enumerate(zip(preds_list, labels_list)):
            loss_walk = dict()
            # loss_walk['loss_walk'] = torch.diagonal(preds, dim1=1,
            #                                         dim2=2).log().mean()
            preds = preds.reshape(-1, preds.size(-1))
            labels = labels.reshape(-1)
            losses.update(
                add_suffix(
                    super().loss(torch.log(preds + 1e-20), labels),
                    suffix=str(idx)))
            losses.update(add_suffix(loss_walk, suffix=str(idx)))

        return losses
