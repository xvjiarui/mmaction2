import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from ..builder import build_loss
from ..registry import HEADS


@HEADS.register_module()
class MoCoHead(nn.Module):
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
                 conv_out_channels=128,
                 num_convs=0,
                 kernel_size=1,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=None,
                 act_cfg=None,
                 num_fcs=2,
                 fc_out_channels=128,
                 with_norm=True,
                 loss_feat=dict(type='MultiPairNCE'),
                 dropout_ratio=0,
                 init_std=0.01,
                 temperature=1.,
                 spatial_type='avg',
                 multi_pair=True,
                 intra_batch=True,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_norm = with_norm
        if loss_feat is None:
            self.loss_feat = None
        else:
            self.loss_feat = build_loss(loss_feat)
        last_layer_dim = in_channels
        if num_convs > 0:
            convs = []
            if num_convs > 1:
                convs.append(
                    ConvModule(
                        self.in_channels,
                        self.conv_out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                for i in range(num_convs - 2):
                    convs.append(
                        ConvModule(
                            self.conv_out_channels,
                            self.conv_out_channels,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
            convs.append(
                ConvModule(
                    self.conv_out_channels
                    if num_convs > 1 else self.in_channels,
                    self.conv_out_channels,
                    kernel_size=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=None,
                    act_cfg=None))
            self.convs = nn.Sequential(*convs)
            last_layer_dim = self.conv_out_channels
        else:
            self.convs = None

        if num_fcs > 1:
            fcs = []
            for i in range(num_fcs - 1):
                fcs.append(nn.Linear(last_layer_dim, self.fc_out_channels))
                fcs.append(nn.ReLU())
                last_layer_dim = self.fc_out_channels
            fcs.append(nn.Linear(last_layer_dim, self.fc_out_channels))
            self.fcs = nn.Sequential(*fcs)
        else:
            self.fcs = None

        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.temperature = temperature
        assert spatial_type in ['avg', None]
        self.spatial_type = spatial_type
        self.multi_pair = multi_pair
        self.intra_batch = intra_batch
        if not intra_batch:
            assert not multi_pair
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
        if self.fcs is not None:
            for fc in self.fcs:
                if isinstance(fc, nn.Linear):
                    xavier_init(fc, distribution='uniform')

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
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        if self.fcs is not None:
            if x.ndim > 2:
                x = x.flatten(1)
            x = self.fcs(x)
        if self.with_norm:
            x = F.normalize(x, p=2, dim=1)
        return x

    def loss(self, query, key, queue):
        """

        Args:
            query (torch.Tensor): [N, C, T]
            key (torch.Tensor): [N, C, T]
            queue (torch.Tensor): [C, K]

        Returns:
            dict:
        """

        batches, channels, clip_len = query.size()
        queue_size = queue.size(1)
        assert key.shape == query.shape
        assert queue.size(0) == channels
        # [NxT, C]
        query = query.transpose(1, 2).reshape(batches * clip_len, channels)
        # [NxT, C]
        key = key.transpose(1, 2).reshape(batches * clip_len, channels)
        if self.intra_batch:
            # [NxT, NxT+K] <- [NxT, C] * [C, NxT+K]
            logits = torch.mm(query, torch.cat([key.T, queue], dim=1))
        else:
            # [NxT, K+1] <- [NxT, C] * [C, K+1]
            logits = torch.cat([
                torch.einsum('nc,nc->n', [query, key]).unsqueeze(-1),
                torch.mm(query, queue)
            ],
                               dim=1)

        if self.multi_pair:
            # [NxT, NxT]
            key_labels = torch.block_diag(
                *[logits.new_ones((clip_len, clip_len), dtype=torch.long)] *
                batches)
        else:
            if self.intra_batch:
                # [NxT, NxT]
                key_labels = torch.eye(
                    batches * clip_len, device=key.device, dtype=torch.long)
            else:
                # [NxT, 1]
                key_labels = torch.ones(
                    batches * clip_len, 1, device=key.device, dtype=torch.long)
        queue_labels = logits.new_zeros((batches * clip_len, queue_size),
                                        dtype=torch.long)
        # [NxT, NxT+K] or [NxT, 1+K]
        labels = torch.cat([key_labels, queue_labels], dim=1)

        # # compute logits
        # # Einstein sum is more intuitive
        # # positive logits: [N, 1]
        # logits_pos = torch.einsum('nc,nc->n', query, key).unsqueeze(1)
        # # negative logits: [N, K]
        # logits_neg = torch.einsum('nc,ck->nkt', query, queue)
        # # intra batch logits: [N, T, N, T]
        # logits_intra = torch.einsum('ncq,nck->nqk', query, key)
        #
        # # logits: [N, (1+K+T), T]
        # logits = torch.cat([logits_pos, logits_neg, logits_intra], dim=1)

        # apply temperature
        logits /= self.temperature

        losses = dict()
        if self.loss_feat is not None:
            # l_pos = torch.einsum('nc,bc->nb', [query, key])
            # # negative logits: NxK
            # l_neg = torch.einsum('nc,ck->nk', [query, queue])
            #
            # # logits: Nx(1+K)
            # logits_x = torch.cat([l_pos, l_neg], dim=1) / self.temperature
            # # labels: positive key indicators
            # labels_x = torch.arange(logits.shape[0], dtype=torch.long,
            #                         device=logits.device)
            #
            # loss_x = self.xent(logits_x, labels_x)
            losses['loss_nce'] = self.loss_feat(logits, labels)
        return losses
