import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import NORM_LAYERS, PLUGIN_LAYERS
from mmcv.cnn import ConvModule as _ConvModule
from mmcv.cnn import build_norm_layer, constant_init


@NORM_LAYERS.register_module(name='SLN')
class SingletonLayerNorm(nn.LayerNorm):

    def __init__(self, num_features, channel_dim=1, **kwargs):
        self.channel_dim = channel_dim
        super(SingletonLayerNorm, self).__init__(
            normalized_shape=num_features, **kwargs)

    def forward(self, x):
        x = x.transpose(self.channel_dim, -1)
        x = super(SingletonLayerNorm, self).forward(x)
        x = x.transpose(self.channel_dim, -1)
        return x

    def extra_repr(self):
        return f'channel_dim={self.channel_dim}'


class ConvModule(_ConvModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 *,
                 norm_cfg=None,
                 **kwargs):
        with_layer_norm = norm_cfg is not None and norm_cfg['type'] == 'LN'
        if with_layer_norm:
            norm_cfg = norm_cfg.copy()
            normalized_shape = norm_cfg.pop('normalized_shape')
        super(ConvModule, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg,
            **kwargs)
        # build normalization layers
        if with_layer_norm:
            # norm layer is after conv layer
            delattr(self, self.norm_name)
            self.norm_name, norm = build_norm_layer(self.norm_cfg,
                                                    normalized_shape)
            self.add_module(self.norm_name, norm)


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
                 normalize=False,
                 matmul_norm=False,
                 dropout=0.0,
                 downsample=None,
                 norm_only=False,
                 detach_key=False):
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
        self.matmul_norm = matmul_norm
        self.detach_key = detach_key
        mid_channels = in_channels // reduction
        out_channels = in_channels

        last_channels = in_channels
        convs = []
        for i in range(num_convs):
            is_last = i == num_convs - 1
            out_channels = out_channels if is_last else \
                mid_channels
            if norm_only:
                convs.append(build_norm_layer(norm_cfg, last_channels)[1])
            else:
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
        self.dropout = nn.Dropout(dropout)
        if downsample is not None:
            assert downsample > 1
            self.downsample3d = nn.MaxPool3d(
                kernel_size=(1, downsample, downsample),
                stride=(1, downsample, downsample),
                ceil_mode=True)
        else:
            self.downsample3d = None

    def downsample_input(self, x):
        if self.downsample3d is None:
            return x
        add_dim = False
        if x.ndim == 4:
            add_dim = True
            # [N, C, 1, H, W]
            x = x.unsqueeze(2)
        x = self.downsample3d(x)
        if add_dim:
            # [N, C, H, W]
            x = x.squeeze(2)
        return x

    def forward(self, query, key=None, value=None):
        if key is None:
            key = query
        if self.detach_key:
            key = key.detach()
        if value is None:
            value = key
        key = self.downsample_input(key)
        value = self.downsample_input(value)
        assert key.shape[2:] == value.shape[2:]
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
        if self.matmul_norm:
            affinity = (query.shape[1]**-.5) * affinity
        affinity = affinity.softmax(dim=1)
        out = torch.matmul(value, affinity).contiguous()
        out = out.view_as(identity)

        scale = out.new_ones((out.size(0), 1, 1, 1))
        scale = self.dropout(scale)
        out *= scale
        if not self.use_residual:
            out += identity * (1 - scale)

        if self.use_residual:
            return out + identity
        else:
            return out


class _SelfAttentionBlock(nn.Module):
    """General self-attention block/non-local block.

    Please refer to https://arxiv.org/abs/1706.03762 for details about key,
    query and value.

    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, key_in_channels, query_in_channels, value_in_channels,
                 channels, out_channels, share_key_query, query_downsample,
                 key_downsample, key_query_num_convs, value_out_num_convs,
                 key_query_norm, value_out_norm, matmul_norm, with_out,
                 conv_cfg, norm_cfg, act_cfg):
        super(_SelfAttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.value_in_channels = value_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.key_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=key_query_num_convs,
            use_conv_module=key_query_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.value_project = self.build_project(
            value_in_channels,
            channels if with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module,
                      conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats, value_feats):
        """Forward function."""
        assert key_feats.shape[2:] == value_feats.shape[2:]
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)
        value = self.value_project(value_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context


@PLUGIN_LAYERS.register_module()
class SelfAttentionBlock(_SelfAttentionBlock):
    """General self-attention block/non-local block.

    Please refer to https://arxiv.org/abs/1706.03762 for details about key,
    query and value.

    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self,
                 query_in_channels,
                 channels,
                 out_channels=None,
                 key_in_channels=None,
                 value_in_channels=None,
                 key_query_num_convs=1,
                 value_out_num_convs=1,
                 key_query_norm=False,
                 value_out_norm=False,
                 share_key_query=False,
                 matmul_norm=True,
                 with_out=True,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 use_residual=True,
                 zero_init=True,
                 dropout=0.0,
                 downsample=None):
        if out_channels is None:
            out_channels = query_in_channels
        if key_in_channels is None:
            key_in_channels = query_in_channels
        if value_in_channels is None:
            value_in_channels = key_in_channels
        self.use_residual = use_residual
        self.zero_init = zero_init
        super(SelfAttentionBlock, self).__init__(
            key_in_channels=key_in_channels,
            query_in_channels=query_in_channels,
            value_in_channels=value_in_channels,
            channels=channels,
            out_channels=out_channels,
            share_key_query=share_key_query,
            key_query_num_convs=key_query_num_convs,
            value_out_num_convs=value_out_num_convs,
            key_query_norm=key_query_norm,
            value_out_norm=value_out_norm,
            matmul_norm=matmul_norm,
            with_out=with_out,
            key_downsample=None,
            query_downsample=None,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.dropout = nn.Dropout2d(dropout)
        if downsample is not None:
            assert downsample > 1
            self.downsample3d = nn.MaxPool3d(
                kernel_size=(1, downsample, downsample),
                stride=(1, downsample, downsample),
                ceil_mode=True)
        else:
            self.downsample3d = None

        # force overwrite
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=1,
                use_conv_module=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        # call init again
        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.zero_init:
            if self.out_project is not None:
                if isinstance(self.out_project, ConvModule):
                    if self.out_project.with_norm:
                        constant_init(self.out_project.norm, 0)
                    else:
                        constant_init(self.out_project.conv, 0)
                else:
                    constant_init(self.out_project, 0)
            elif self.value_project is not None:
                if isinstance(self.value_project, ConvModule):
                    if self.value_project.with_norm:
                        constant_init(self.value_project.norm, 0)
                    else:
                        constant_init(self.value_project.conv, 0)
                else:
                    constant_init(self.value_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module,
                      conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if num_convs == 0:
            return nn.Identity()
        if use_conv_module:
            convs = [
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        else:
            convs = [nn.Conv1d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv1d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def downsample_input(self, x):
        if self.downsample3d is None:
            return x
        add_dim = False
        if x.ndim == 4:
            add_dim = True
            # [N, C, 1, H, W]
            x = x.unsqueeze(2)
        x = self.downsample3d(x)
        if add_dim:
            # [N, C, H, W]
            x = x.squeeze(2)
        return x

    def forward(self, query_feats, key_feats=None, value_feats=None):
        if key_feats is None:
            key_feats = query_feats
        if value_feats is None:
            value_feats = key_feats
        key_feats = self.downsample_input(key_feats)
        value_feats = self.downsample_input(value_feats)
        out = super().forward(
            query_feats.flatten(2), key_feats.flatten(2),
            value_feats.flatten(2))
        out = out.reshape_as(query_feats)
        out = self.dropout(out)
        if self.use_residual:
            out = out + query_feats
        return out


@PLUGIN_LAYERS.register_module()
class MultiHeadAttention(nn.Module):
    """A warpper for torch.nn.MultiheadAttention.

    This module implements MultiheadAttention with residual connection,
    and positional encoding used in DETR is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        dropout (float): A Dropout layer on attn_output_weights. Default 0.0.
        batchwise_drop (float): Drop attention batchwisely. Default: False
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 dropout=0.0,
                 batchwise_drop=False,
                 use_residual=True):
        super(MultiHeadAttention, self).__init__()
        assert embed_dims % num_heads == 0, \
            f'embed_dims must be divisible by num_heads. got {embed_dims} ' \
            f'and {num_heads}.'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout if not batchwise_drop else 0.)
        self.dropout = nn.Dropout(dropout)
        self.batchwise_drop = batchwise_drop
        self.use_residual = use_residual

    def forward(self, x, key=None, value=None):
        """Forward function for `MultiheadAttention`.

        Args:
            x (Tensor): The input query with shape [num_query, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
            key (Tensor): The key tensor with shape [num_key, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
                Default None. If None, the `query` will be used.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Default None.
                If None, the `key` will be used.

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        query = x
        if key is None:
            key = query
        if value is None:
            value = key
        assert key.shape[2:] == value.shape[2:]
        query = query.flatten(2).permute(2, 0,
                                         1)  # [bs, c, h, w] -> [h*w, bs, c]
        key = key.flatten(2).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        value = value.flatten(2).permute(2, 0,
                                         1)  # [bs, c, h, w] -> [h*w, bs, c]
        out, out_weights = self.attn(
            query, key, value=value, attn_mask=None, key_padding_mask=None)
        if self.batchwise_drop:
            scale = out.new_ones((1, out.size(1), 1))
            scale = self.dropout(scale)
            out *= scale
            if not self.use_residual:
                out += query * (1 - scale)
        else:
            out = self.dropout(out)
        out = out.permute(1, 2, 0).reshape_as(x)
        if self.use_residual:
            out = x + out

        return out

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(embed_dims={self.embed_dims}, '
        repr_str += f'num_heads={self.num_heads}, '
        repr_str += f'dropout={self.dropout})'
        return repr_str


@PLUGIN_LAYERS.register_module()
class AttentionPool2d(nn.Module):

    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1],
                      x.shape[2] * x.shape[3]).permute(2, 0,
                                                       1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)

        return x[0]
