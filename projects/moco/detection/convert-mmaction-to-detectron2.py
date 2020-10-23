import os.path as osp
import pickle as pkl
import argparse
from collections import OrderedDict

import torch


def convert(src, dst):
    """Convert keys in detectron pretrained ResNet models to pytorch style."""
    # convert to pytorch style
    print(f'src: {osp.realpath(src)}')
    print('='*10, 'converting to torchvision', '='*10, '\n')
    state_dict = OrderedDict()
    src_dict = torch.load(src, map_location='cpu')
    src_state_dict = src_dict.get('state_dict', src_dict)
    for k, v in src_state_dict.items():
        if not k.startswith('backbone'):
            continue
        b_k = k.replace('backbone.', '')
        b_k_splits = b_k.split('.')
        tail = b_k_splits[-1]
        if b_k.startswith('conv1'):
            if b_k_splits[1] == 'conv':
                name = f'conv1.{tail}'
            elif b_k_splits[1] == 'bn':
                name = f'bn1.{tail}'
            else:
                raise RuntimeError(b_k)
        elif b_k.startswith('layer'):
            layer_idx = int(b_k_splits[0][-1])
            block_idx = int(b_k_splits[1])
            if b_k_splits[2] == 'downsample':
                # downsample
                if b_k_splits[3] == 'conv':
                    name = f'layer{layer_idx}.{block_idx}.downsample.0.{tail}'
                elif b_k_splits[3] == 'bn':
                    name = f'layer{layer_idx}.{block_idx}.downsample.1.{tail}'
                else:
                    raise RuntimeError(b_k)
            elif b_k_splits[3] == 'conv':
                conv_module_idx = int(b_k_splits[2][-1])
                name = f'layer{layer_idx}.{block_idx}.' \
                       f'conv{conv_module_idx}.{tail}'
            elif b_k_splits[3] == 'bn':
                conv_module_idx = int(b_k_splits[2][-1])
                name = f'layer{layer_idx}.{block_idx}.' \
                       f'bn{conv_module_idx}.{tail}'
            else:
                raise RuntimeError(b_k)
        else:
            raise RuntimeError(f'{b_k}')
        state_dict[name] = v
        print(f'{k} --> {name}')

    print('='*10, 'converting to detectron2', '='*10, '\n')

    newmodel = {}
    obj = state_dict
    for k in list(obj.keys()):
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "MMAction2", "matching_heuristics": True}

    with open(dst, "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())

def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src mmaction model path')
    parser.add_argument('dst', help='save detectron2 path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
