import argparse
from collections import OrderedDict

import torch

conversion_pairs = {'encoder.model.': ''}


def convert(src, dst):
    """Convert keys in detectron pretrained ResNet models to pytorch style."""
    # convert to pytorch style
    state_dict = OrderedDict()
    src_dict = torch.load(src, map_location='cpu')
    src_state_dict = src_dict['model']
    for k, v in src_state_dict.items():
        converted = False
        for src_name, dst_name in conversion_pairs.items():
            if k.startswith(src_name):
                # print('{} is converted'.format(k))
                if k.replace(src_name, dst_name) in state_dict:
                    print('{} is duplicate'.format(k))
                    continue
                state_dict[k.replace(src_name, dst_name)] = v
                print(f'{k}->{k.replace(src_name, dst_name)}')
                converted = True
                break
        if not converted:
            print('{} not converted'.format(k))
            state_dict[k] = v

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    assert len(state_dict) == len(src_state_dict), '{} vs {}'.format(
        len(state_dict), len(src_state_dict))
    checkpoint['meta'] = dict()
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
