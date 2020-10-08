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
    batches, channels, clip_len = imgs.shape[:3]
    new_imgs = imgs.transpose(1,
                              2).contiguous().reshape(batches * clip_len,
                                                      channels,
                                                      *imgs.shape[3:])

    return new_imgs


def images2video(imgs, clip_len):
    batches, channels = imgs.shape[:2]
    new_imgs = imgs.reshape(batches // clip_len, clip_len, channels,
                            *imgs.shape[2:]).transpose(1, 2).contiguous()

    return new_imgs


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class StrideContext(object):

    def __init__(self, backbone, strides):
        self.backbone = backbone
        self.strides = strides

    def __enter__(self):
        if self.strides is not None:
            self.backbone.switch_strides(self.strides)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.strides is not None:
            self.backbone.switch_strides()
