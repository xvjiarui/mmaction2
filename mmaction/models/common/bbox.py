from typing import Tuple

import kornia.geometry as T
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def center2bbox(center, patch_size, img_size):
    patch_size = _pair(patch_size)
    img_size = _pair(img_size)
    patch_h, patch_w = patch_size
    img_h, img_w = img_size
    new_l = center[:, 0] - patch_w / 2
    new_l[new_l < 0] = 0
    new_l = new_l.unsqueeze(dim=1)

    new_r = new_l + patch_w
    new_r[new_r > img_w] = img_w

    new_t = center[:, 1] - patch_h / 2
    new_t[new_t < 0] = 0
    new_t = new_t.unsqueeze(dim=1)

    new_b = new_t + patch_h
    new_b[new_b > img_h] = img_h

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
    # src = diff_crop(tensor, boxes, size[0], size[1])
    # dst = T.crop_and_resize(tensor=tensor, boxes=complete_bboxes(boxes),
    #                         size=size,
    #                         interpolation=interpolation,
    #                         align_corners=align_corners)
    # print((src - dst).abs().max())
    # TODO check
    # return diff_crop(tensor, boxes, size[0], size[1])
    boxes = complete_bboxes(boxes)
    return T.crop_and_resize(
        tensor=tensor,
        boxes=boxes,
        size=size,
        interpolation=interpolation,
        align_corners=align_corners)


def diff_crop(imgs, bboxes, out_height, out_width):
    """
    Differatiable cropping
    INPUTS:
     - F: frame feature
     - x1,y1,x2,y2: top left and bottom right points of the patch
     - theta is defined as :
                        a b c
                        d e f
    """
    assert imgs.size(0) == bboxes.size(0)
    assert bboxes.size(-1) == 4
    x1, y1, x2, y2 = bboxes.split(1, dim=1)
    batches, channels, height, width = imgs.size()
    a = ((x2 - x1) / width).view(batches, 1, 1)
    b = torch.zeros(a.size()).cuda()
    c = (-1 + (x1 + x2) / width).view(batches, 1, 1)
    d = torch.zeros(a.size()).cuda()
    e = ((y2 - y1) / height).view(batches, 1, 1)
    f = (-1 + (y2 + y1) / height).view(batches, 1, 1)
    theta_row1 = torch.cat((a, b, c), dim=2)
    theta_row2 = torch.cat((d, e, f), dim=2)
    theta = torch.cat((theta_row1, theta_row2), dim=1).cuda()
    grid = F.affine_grid(
        theta,
        size=torch.Size((batches, channels, out_width, out_height)),
        align_corners=False)
    patch = F.grid_sample(imgs, grid, align_corners=False)
    return patch


def get_random_crop_bbox(batches, crop_size, img_size, device):
    """Randomly get a crop bounding box."""
    crop_size = _pair(crop_size)
    img_size = _pair(img_size)
    margin_h = max(img_size[0] - crop_size[0], 0)
    margin_w = max(img_size[1] - crop_size[1], 0)
    offset_h = torch.randint(margin_h + 1, size=(batches, 1), device=device)
    offset_w = torch.randint(margin_w + 1, size=(batches, 1), device=device)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
    bbox = torch.cat([crop_x1, crop_y1, crop_x2, crop_y2], dim=1).float()

    return bbox
