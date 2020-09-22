from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def center2bbox(center, patch_size, img_size):
    patch_size = _pair(patch_size)
    img_size = _pair(img_size)
    patch_h, patch_w = patch_size
    img_h, img_w = img_size
    new_l = center[:, 0] - patch_w / 2
    new_l = new_l.clamp(min=0)
    new_l = new_l.unsqueeze(dim=1)

    new_r = new_l + patch_w
    new_r = new_r.clamp(max=img_w)

    new_t = center[:, 1] - patch_h / 2
    new_t = new_t.clamp(min=0)
    new_t = new_t.unsqueeze(dim=1)

    new_b = new_t + patch_h
    new_b = new_b.clamp(max=img_h)

    bboxes = torch.cat((new_l, new_t, new_r, new_b), dim=1)
    return bboxes


def complete_bboxes(bboxes):
    x1, y1, x2, y2 = bboxes.split(1, dim=1)
    tl = torch.cat([x1, y1], dim=1)
    tr = torch.cat([x2, y1], dim=1)
    bl = torch.cat([x1, y2], dim=1)
    br = torch.cat([x2, y2], dim=1)

    return torch.stack([tl, tr, br, bl], dim=1)


def crop_and_resize(tensor: torch.Tensor,
                    boxes: torch.Tensor,
                    size: Tuple[int, int],
                    interpolation: str = 'bilinear',
                    align_corners: bool = False) -> torch.Tensor:
    # src = _crop_and_resize(tensor, boxes, size[0], size[1])
    # dst = T.crop_and_resize(tensor=tensor, boxes=complete_bboxes(boxes),
    #                         size=size,
    #                         interpolation=interpolation,
    #                         align_corners=align_corners)
    # print((src - dst).abs().max())
    # TODO check
    return _crop_and_resize(tensor, boxes, size, interpolation, align_corners)
    # boxes = complete_bboxes(boxes)
    # return T.crop_and_resize(
    #     tensor=tensor,
    #     boxes=boxes,
    #     size=size,
    #     interpolation=interpolation,
    #     align_corners=align_corners)


def get_crop_grid(img,
                  bboxes,
                  out_size,
                  device=None,
                  dtype=None,
                  align_corners=False):
    """theta is defined as :

    a b c d e f
    """
    if isinstance(img, torch.Tensor):
        img_shape = img.shape
    elif isinstance(img, torch.Size):
        img_shape = img
    else:
        raise RuntimeError('img must be Tensor or Size')
    device = img.device if device is None else device
    dtype = img.dtype if dtype is None else dtype
    assert img_shape[0] == bboxes.size(0)
    assert bboxes.size(-1) == 4
    x1, y1, x2, y2 = bboxes.split(1, dim=1)
    batches, channels, height, width = img_shape
    a = ((x2 - x1) / width).view(batches, 1, 1)
    b = torch.zeros(*a.size(), device=device, dtype=dtype)
    c = (-1 + (x1 + x2) / width).view(batches, 1, 1)
    d = torch.zeros(*a.size(), device=device, dtype=dtype)
    e = ((y2 - y1) / height).view(batches, 1, 1)
    f = (-1 + (y2 + y1) / height).view(batches, 1, 1)
    theta_row1 = torch.cat((a, b, c), dim=2)
    theta_row2 = torch.cat((d, e, f), dim=2)
    theta = torch.cat((theta_row1, theta_row2), dim=1)
    grid = F.affine_grid(
        theta,
        size=torch.Size((batches, channels, *out_size)),
        align_corners=align_corners)
    return grid


def _crop_and_resize(imgs, bboxes, out_size, interpolation, align_corners):
    grid = get_crop_grid(imgs, bboxes, out_size, align_corners=align_corners)
    patch = F.grid_sample(imgs, grid, mode=interpolation, align_corners=False)
    return patch


def get_random_crop_bbox(batches,
                         crop_size,
                         img_size,
                         device,
                         center_ratio=0.):
    """Randomly get a crop bounding box."""
    crop_size = _pair(crop_size)
    img_size = _pair(img_size)
    margin_h = max(img_size[0] - crop_size[0], 0)
    margin_w = max(img_size[1] - crop_size[1], 0)

    if np.random.rand() < center_ratio:
        center_crop = True
    else:
        center_crop = False
    if center_crop:
        offset_h = torch.full(
            fill_value=(margin_h + 1) // 2,
            size=(batches, 1),
            device=device,
            dtype=torch.int)
        offset_w = torch.full(
            fill_value=(margin_w + 1) // 2,
            size=(batches, 1),
            device=device,
            dtype=torch.int)
    else:
        offset_h = torch.randint(
            margin_h + 1, size=(batches, 1), device=device)
        offset_w = torch.randint(
            margin_w + 1, size=(batches, 1), device=device)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
    bbox = torch.cat([crop_x1, crop_y1, crop_x2, crop_y2], dim=1).float()

    return bbox, center_crop
