import torch
import torch.nn.functional as F


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


def transform(aff, img):
    b, c, h, w = img.size()
    img = img.view(b, c, -1)
    frame2 = torch.bmm(img, aff)
    return frame2.view(b, c, h, w)


def propagate(img, affinity, topk=None):
    batches, channels, height, width = img.size()
    if topk is not None:
        tk_val, tk_idx = affinity.topk(dim=1, k=topk)
        tk_val_min, _ = tk_val.min(dim=1)
        tk_val_min = tk_val_min.view(batches, 1, height * width)
        affinity[tk_val_min > affinity] = 0
    img = img.view(batches, channels, -1)
    new_img = torch.bmm(img, affinity)
    new_img = new_img.view(batches, channels, height, width)
    return new_img
