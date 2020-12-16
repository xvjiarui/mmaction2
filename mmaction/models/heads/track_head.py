import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops.point_sample import generate_grid

from ..builder import build_loss
from ..common import center2bbox, compute_affinity, coord2bbox
from ..registry import HEADS


@HEADS.register_module()
class TrackHead(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_feat (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 num_convs=0,
                 kernel_size=1,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 normalize=True,
                 loss_grid=dict(type='MSELoss'),
                 loss_aff=None,
                 temperature=1.,
                 track_type='center'):
        super().__init__()
        self.in_channels = in_channels
        self.num_convs = num_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.normalize = normalize
        self.loss_grid = build_loss(loss_grid)
        if loss_aff is not None:
            self.loss_aff = build_loss(loss_aff)
        else:
            self.loss_aff = None
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

        self.temperature = temperature
        assert track_type in ['center', 'coord']
        self.track_type = track_type

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def estimate_grid(self, x, pred_x):
        # [N, w*h, 2]
        grid = generate_grid(x.shape[0], x.shape[2:], device=x.device)
        affinity = compute_affinity(
            x,
            pred_x,
            softmax_dim=None,
            temperature=self.temperature,
            normalize=self.normalize).contiguous()
        # [N, w*h, 2]
        pred_grid = torch.bmm(affinity.softmax(dim=2), grid)
        cycle_grid = torch.bmm(
            affinity.transpose(1, 2).softmax(dim=2), pred_grid)

        grid = grid.transpose(2, 1).reshape(x.shape[0], 2,
                                            *x.shape[2:]).contiguous()
        cycle_grid = cycle_grid.transpose(2, 1).reshape(
            x.shape[0], 2, *x.shape[2:]).contiguous()

        return affinity, grid, cycle_grid

    def estimate_bbox(self, patch_x, x):
        # [N, x_w*x_h, 2]
        x_grid = generate_grid(x.shape[0], x.shape[2:], device=x.device)
        # [N, x_w*x_h, 2]
        x_coords = torch.stack(
            [x_grid[..., 0] * x.shape[3], x_grid[..., 1] * x.shape[2]],
            dim=2).contiguous()
        affinity = compute_affinity(
            patch_x,
            x,
            softmax_dim=2,
            temperature=self.temperature,
            normalize=self.normalize).contiguous()
        # [N, patch_w*patch_h, 2]
        pred_coords = torch.bmm(affinity, x_coords)
        # pred_coords_ = masked_attention_efficient(
        #     patch_x, x,
        #     x_coords.permute(0, 2, 1).reshape(x.shape[0], 2, *x.shape[2:]),
        #     mask=None,
        #     normalize=self.normalize,
        #     temperature=self.temperature)
        # pred_coords_ = pred_coords_.view(patch_x.shape[0], 2, -1).
        # permute(0, 2, 1)
        # assert torch.allclose(pred_coords_.detach(), pred_coords.detach())
        if self.track_type == 'coord':
            pred_bboxes = coord2bbox(pred_coords, x.shape[2:])
        else:
            pred_bboxes = center2bbox(
                torch.mean(pred_coords, dim=1), patch_x.shape[2:], x.shape[2:])

        return pred_bboxes

    def forward(self, patch_x, x):
        patch_x = self.convs(patch_x)
        x = self.convs(x)
        pred_bboxes = self.estimate_bbox(patch_x, x)

        return pred_bboxes

    def loss(self, grid1, grid2, affinity, weight=1.):
        assert grid1.shape == grid2.shape
        assert grid1.shape[1] == grid2.shape[1] == 2

        losses = dict()

        if self.loss_aff is not None:
            loss_aff = self.loss_aff(affinity)
            losses['loss_aff'] = loss_aff * weight

        loss_grid = self.loss_grid(grid1, grid2)
        losses['loss_grid'] = loss_grid * weight
        return losses
