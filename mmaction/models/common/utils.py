import mmcv
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair, _single, _triple


def change_stride(conv, stride):
    """Inplace change conv stride.

    Args:
        conv (nn.Module):
        stride (int):
    """
    if isinstance(conv, nn.Conv1d):
        conv.stride = _single(stride)
    if isinstance(conv, nn.Conv2d):
        conv.stride = _pair(stride)
    if isinstance(conv, nn.Conv3d):
        conv.stride = _triple(stride)


def pil_nearest_interpolate(input, size):
    # workaround for https://github.com/pytorch/pytorch/issues/34808
    resized_imgs = []
    input = input.permute(0, 2, 3, 1)
    for img in input:
        img = img.squeeze(-1)
        img = img.detach().cpu().numpy()
        resized_img = mmcv.imresize(
            img,
            size=(size[1], size[0]),
            interpolation='nearest',
            backend='pillow')
        resized_img = torch.from_numpy(resized_img).to(
            input, non_blocking=True)
        resized_img = resized_img.unsqueeze(2).permute(2, 0, 1)
        resized_imgs.append(resized_img)

    return torch.stack(resized_imgs, dim=0)


def video2images(imgs):
    batches, channels, clip_len, height, width = imgs.size()
    new_imgs = imgs.transpose(1,
                              2).contiguous().reshape(batches * clip_len,
                                                      channels, height, width)

    return new_imgs


def images2video(imgs, clip_len):
    batches, channels, height, width = imgs.size()
    new_imgs = imgs.reshape(batches // clip_len, clip_len, channels, height,
                            width).transpose(1, 2).contiguous()

    return new_imgs
