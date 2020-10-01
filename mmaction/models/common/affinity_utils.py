import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def compute_affinity(src_img,
                     dst_img,
                     temperature=1.,
                     normalize=True,
                     softmax_dim=None):
    batches, channels = src_img.shape[:2]
    src_feat = src_img.view(batches, channels, src_img.shape[2:].numel())
    dst_feat = dst_img.view(batches, channels, dst_img.shape[2:].numel())
    if normalize:
        src_feat = F.normalize(src_feat, p=2, dim=1)
        dst_feat = F.normalize(dst_feat, p=2, dim=1)
    src_feat = src_feat.permute(0, 2, 1).contiguous()
    dst_feat = dst_feat.contiguous()
    affinity = torch.bmm(src_feat, dst_feat) / temperature
    if softmax_dim is not None:
        affinity = affinity.softmax(dim=softmax_dim)

    return affinity


def propagate(img, affinity, topk=None, mask=None):
    batches, channels, height, width = img.size()
    if topk is not None:
        tk_val, tk_idx = affinity.topk(dim=1, k=topk)
        tk_val_min, _ = tk_val.min(dim=1)
        tk_val_min = tk_val_min.view(batches, 1, height * width)
        affinity[tk_val_min > affinity] = 0
    if mask is not None:
        affinity *= mask
    img = img.view(batches, channels, -1)
    img = img.contiguous()
    affinity = affinity.contiguous()
    new_img = torch.bmm(img, affinity).contiguous()
    new_img = new_img.reshape(batches, channels, height, width)
    return new_img


def spatial_neighbor(batches, height, width, neighbor_range, device, dtype):
    neighbor_range = _pair(neighbor_range)
    mask = torch.zeros(
        batches, height, width, height, width, device=device, dtype=dtype)
    for i in range(height):
        for j in range(width):
            top = max(0, i - neighbor_range[0] // 2)
            left = max(0, j - neighbor_range[1] // 2)
            bottom = min(height, i + neighbor_range[0] // 2 + 1)
            right = min(width, j + neighbor_range[1] // 2 + 1)
            mask[:, top:bottom, left:right, i, j] = 1

    mask = mask.view(batches, height * width, height * width)
    return mask
