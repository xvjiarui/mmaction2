import warnings

import torch
import torch.nn as nn
from mmcv.cnn import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module()
class AvgFusion(nn.Module):

    def __init__(self, avg_dim=None):
        super().__init__()
        self.avg_dim = avg_dim

    def forward(self, x1, x2):
        if x1.shape == x2.shape:
            warnings.warn(f'self.avg_dim={self.avg_dim} has no effect')
            return (x1 + x2) * 0.5
        assert self.avg_dim < x2.ndim, f'{self.avg_dim} < {x2.ndim}'
        avg_num = x2.size(self.avg_dim)
        x1 = x1 + x2.sum(dim=self.avg_dim)
        x1 = x1 / (avg_num + 1)

        return x1

    def extra_repr(self):
        return f'avg_dim={self.avg_dim}'


@PLUGIN_LAYERS.register_module()
class CatFusion(nn.Module):

    def __init__(self, cat_dim=1):
        super().__init__()
        self.cat_dim = cat_dim

    def forward(self, x1, x2):
        if x1.shape == x2.shape:
            warnings.warn('concat on dim 1')
            return torch.cat((x1, x2), dim=1)
        assert self.cat_dim + 1 < x2.ndim, f'{self.cat_dim+1} < {x2.ndim}'
        x1 = torch.cat((x1.unsqueeze(dim=self.cat_dim + 1), x2),
                       dim=self.cat_dim)
        x1 = x1.flatten(start_dim=self.cat_dim, end_dim=self.cat_dim + 1)

        return x1

    def extra_repr(self):
        return f'cat_dim={self.cat_dim}'
