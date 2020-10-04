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


def coord2bbox(coords, img_size):
    img_size = _pair(img_size)
    img_h, img_w = img_size
    batches = coords.size(0)
    # [N,  2]
    center = torch.mean(coords, dim=1)
    center_repeat = center.unsqueeze(1).repeat(1, coords.size(1), 1)
    dis_x = torch.abs(coords[:, :, 0] - center_repeat[:, :, 0])
    dis_x = torch.mean(dis_x, dim=1).detach()
    dis_y = torch.abs(coords[:, :, 1] - center_repeat[:, :, 1])
    dis_y = torch.mean(dis_y, dim=1).detach()
    left = (center[:, 0] - dis_x * 2).view(batches, 1)
    left = left.clamp(min=0)
    right = (center[:, 0] + dis_x * 2).view(batches, 1)
    right = right.clamp(max=img_w)
    top = (center[:, 1] - dis_y * 2).view(batches, 1)
    top = top.clamp(min=0)
    bottom = (center[:, 1] + dis_y * 2).view(batches, 1)
    bottom = bottom.clamp(max=img_h)
    bboxes = torch.cat((left, top, right, bottom), dim=1)
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
    assert bboxes.size(1) == 4
    assert bboxes.ndim == 2
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
                         center_ratio=0.,
                         border=0):
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
            low=border,
            high=margin_h + 1 - border,
            size=(batches, 1),
            device=device)
        offset_w = torch.randint(
            low=border,
            high=margin_w + 1 - border,
            size=(batches, 1),
            device=device)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
    bbox = torch.cat([crop_x1, crop_y1, crop_x2, crop_y2], dim=1).float()

    return bbox, center_crop


def get_top_diff_crop_bbox(imgs,
                           ref_imgs,
                           crop_size,
                           grid_size,
                           device,
                           topk=10):
    """Randomly get a crop bounding box."""
    assert imgs.shape == ref_imgs.shape
    batches = imgs.size(0)
    img_size = imgs.shape[2:]
    crop_size = _pair(crop_size)
    grid_size = _pair(grid_size)
    stride_h = (img_size[0] - crop_size[0]) // (grid_size[0] - 1)
    stride_w = (img_size[1] - crop_size[1]) // (grid_size[1] - 1)
    diff_imgs = imgs - ref_imgs

    diff_list = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            crop_diff = diff_imgs[:, :,
                                  i * stride_h:i * stride_h + crop_size[0],
                                  j * stride_w:j * stride_w + crop_size[1]]
            diff_list.append(crop_diff.abs().sum(dim=(1, 2, 3)))
    # [batches, grid_size**2]
    diff_sum = torch.stack(diff_list, dim=1)
    diff_topk_idx = torch.argsort(diff_sum, dim=1, descending=True)[:, :topk]
    perm = torch.randint(low=0, high=topk, size=(batches, ), device=device)
    select_idx = diff_topk_idx[torch.arange(batches, device=device), perm]
    idx_i = select_idx // grid_size[1]
    idx_j = select_idx % grid_size[1]

    crop_y1, crop_y2 = idx_i * stride_h, idx_i * stride_h + crop_size[0]
    crop_x1, crop_x2 = idx_j * stride_w, idx_j * stride_w + crop_size[1]
    bbox = torch.stack([crop_x1, crop_y1, crop_x2, crop_y2], dim=1).float()

    return bbox


@torch.no_grad()
def bbox2mask(bboxes, mask_shape, neighbor_range=0):
    assert bboxes.size(1) == 4
    assert bboxes.ndim == 2
    batches = bboxes.size(0)
    height, width = mask_shape
    mask = bboxes.new_zeros(batches, height, width)
    neighbor_range = _pair(neighbor_range)

    mask_out = []
    # TODO: Looking for a vectorized way
    for m, box in zip(mask, bboxes):
        x1, y1, x2, y2 = box.split(1, dim=0)
        top = max(0, y1.item() - neighbor_range[0] // 2)
        left = max(0, x1.item() - neighbor_range[1] // 2)
        bottom = min(height, y2.item() + neighbor_range[0] // 2 + 1)
        right = min(width, x2.item() + neighbor_range[1] // 2 + 1)
        m = m.index_fill(
            1, torch.arange(left, right, dtype=torch.long, device=box.device),
            torch.tensor(1, dtype=box.dtype, device=box.device))
        m = m.index_fill(
            0, torch.arange(top, bottom, dtype=torch.long, device=box.device),
            torch.tensor(1, dtype=box.dtype, device=box.device))
        m = m.unsqueeze(dim=0)
        m_out = (m == 1).all(dim=1) * (m == 1).all(dim=2).T
        mask_out.append(m_out)

    return torch.stack(mask_out, dim=0).to(dtype=bboxes.dtype)
