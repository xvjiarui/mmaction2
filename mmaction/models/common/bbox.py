import os.path as osp
from typing import Tuple

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.ops.point_sample import generate_grid
from torch.nn.modules.utils import _pair

from .affinity_utils import compute_affinity


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


def get_non_overlap_crop_bbox(ori_bboxes,
                              img_size,
                              radius=1.,
                              scale_jitter=0.):
    """Randomly get a crop bounding box."""
    assert ori_bboxes.size(1) == 4
    assert ori_bboxes.ndim == 2
    batches = ori_bboxes.size(0)
    device = ori_bboxes.device
    dtype = ori_bboxes.dtype
    img_size = _pair(img_size)
    x1 = ori_bboxes[..., 0]
    y1 = ori_bboxes[..., 1]
    x2 = ori_bboxes[..., 2]
    y2 = ori_bboxes[..., 3]
    ori_center_x = (x1 + x2) * 0.5
    ori_center_y = (y1 + y2) * 0.5
    ori_width = x2 - x1
    ori_height = y2 - y1

    new_center_x = torch.randint(
        low=0, high=img_size[1], size=(batches, ), device=device, dtype=dtype)
    new_center_x = torch.where(
        torch.logical_and(
            new_center_x > (ori_center_x - ori_width * 0.5 * radius - 1),
            new_center_x < ori_center_x + 1),
        new_center_x - ori_width * radius + 1, new_center_x)
    new_center_x = torch.where(
        torch.logical_and(
            new_center_x < (ori_center_x + ori_width * 0.5 * radius + 1),
            new_center_x > ori_center_x - 1),
        new_center_x + ori_width * radius + 1, new_center_x)

    new_center_y = torch.randint(
        low=0, high=img_size[0], size=(batches, ), device=device, dtype=dtype)

    new_center_y = torch.where(
        torch.logical_and(
            new_center_y > (ori_center_y - ori_height * 0.5 * radius - 1),
            new_center_y < ori_center_y + 1),
        new_center_y - ori_height * radius + 1, new_center_y)
    new_center_y = torch.where(
        torch.logical_and(
            new_center_y < (ori_center_y + ori_height * 0.5 * radius + 1),
            new_center_y > ori_center_y - 1),
        new_center_y + ori_height * radius + 1, new_center_y)

    if scale_jitter > 0:
        new_width = ori_width * (
            1 + (torch.rand(batches, device=device) * 2 - 1) * scale_jitter)
        new_height = ori_height * (
            1 + (torch.rand(batches, device=device) * 2 - 1) * scale_jitter)
    else:
        new_width = ori_width
        new_height = ori_height

    new_x1 = new_center_x - new_width * 0.5
    new_y1 = new_center_y - new_height * 0.5
    new_x2 = new_center_x + new_width * 0.5
    new_y2 = new_center_y + new_height * 0.5

    new_x1.clamp_(min=0, max=img_size[1])
    new_y1.clamp_(min=0, max=img_size[0])
    new_x2.clamp_(min=0, max=img_size[1])
    new_y2.clamp_(min=0, max=img_size[0])

    new_bboxes = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)

    return new_bboxes


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


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, 1))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])

        if mode == 'iou':
            area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
                bboxes2[..., 3] - bboxes2[..., 1])
            union = area1 + area2 - overlap
        else:
            union = area1
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])

        if mode == 'iou':
            area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
                bboxes2[..., 3] - bboxes2[..., 1])
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union

    return ious


def scale_bboxes(bboxes, img_size, scale=1.):
    assert bboxes.size(1) == 4
    assert bboxes.ndim == 2
    batches = bboxes.size(0)
    x1 = bboxes[..., 0]
    y1 = bboxes[..., 1]
    x2 = bboxes[..., 2]
    y2 = bboxes[..., 3]
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    width = x2 - x1
    height = y2 - y1

    if isinstance(scale, tuple):
        assert scale[1] >= scale[0]
        scale = torch.rand(
            batches, device=bboxes.device) * (scale[1] - scale[0]) + scale[0]

    new_width = width * scale
    new_height = height * scale

    new_x1 = center_x - new_width * 0.5
    new_y1 = center_y - new_height * 0.5
    new_x2 = center_x + new_width * 0.5
    new_y2 = center_y + new_height * 0.5

    new_x1.clamp_(min=0, max=img_size[1])
    new_y1.clamp_(min=0, max=img_size[0])
    new_x2.clamp_(min=0, max=img_size[1])
    new_y2.clamp_(min=0, max=img_size[0])

    new_bboxes = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)

    return new_bboxes


def vis_imgs(img, save_dir='debug_results'):
    mean = torch.tensor([123.675, 116.28, 103.53]).to(img).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).to(img).view(1, 3, 1, 1)
    save_img = img * std + mean
    for save_idx in range(save_img.size(0)):
        img_cur = save_img[save_idx].permute(1, 2, 0).detach().cpu().numpy()
        img_cur = mmcv.rgb2bgr(img_cur)
        mmcv.mkdir_or_exist(save_dir)
        mmcv.imwrite(img_cur, osp.join(save_dir, f'{save_idx}.jpg'))


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def roi2bbox(rois):
    """Convert rois to bounding box format.

    Args:
        rois (torch.Tensor): RoIs with the shape (n, 5) where the first
            column indicates batch id of each RoI.

    Returns:
        list[torch.Tensor]: Converted boxes of corresponding rois.
    """
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (rois[:, 0] == img_id.item())
        bbox = rois[inds, 1:]
        bbox_list.append(bbox)
    return bbox_list


def propagate_bbox(ref_crop_x,
                   tar_x,
                   track_type='center',
                   temperature=1.,
                   with_norm=True):
    assert track_type in ['center', 'coord']
    assert tar_x.ndim in [4, 5]
    if tar_x.ndim == 4:
        tar_shape = tar_x.shape
    else:
        tar_shape = tar_x[:, :, 0].shape
    # [N, tar_w*tar_h, 2]
    tar_grid = generate_grid(
        tar_shape[0], tar_shape[2:], device=ref_crop_x.device)
    # TODO check
    # [N, tar_w*tar_h, 2]
    tar_coords = torch.stack(
        [tar_grid[..., 0] * tar_shape[3], tar_grid[..., 1] * tar_shape[2]],
        dim=2).contiguous()
    # [N, ref_w*ref_h, tar_w*tar_h]
    aff_ref_tar = compute_affinity(
        ref_crop_x,
        tar_x,
        temperature=temperature,
        normalize=with_norm,
        softmax_dim=2).contiguous()

    # [N, ref_w*ref_h, 2]
    ref_coords = torch.bmm(aff_ref_tar, tar_coords)
    if track_type == 'coord':
        tar_bboxes = coord2bbox(ref_coords, tar_shape[2:])
    else:
        # [N, 2]
        ref_center = torch.mean(ref_coords, dim=1)
        tar_bboxes = center2bbox(ref_center, ref_crop_x.shape[2:],
                                 tar_shape[2:])

    return tar_bboxes
