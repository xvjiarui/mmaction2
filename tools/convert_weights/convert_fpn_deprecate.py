import argparse
from collections import OrderedDict

import torch

conversion_pairs = {
    'neck.lateral_convs.0': 'neck.lateral_convs.1',
    'neck.lateral_convs.1': 'neck.lateral_convs.2',
    'neck.lateral_convs.2': 'neck.lateral_convs.3',
}


def convert(src, dst):
    """Convert keys in detectron pretrained ResNet models to pytorch style."""
    # convert to pytorch style
    state_dict = OrderedDict()
    src_dict = torch.load(src)
    src_state_dict = src_dict.get('state_dict', src_dict)
    delete_keys = []
    for k, v in src_state_dict.items():
        if k.startswith('loss'):
            delete_keys.append(k)
            continue
        converted = False
        for src_name, dst_name in conversion_pairs.items():
            if k.startswith(src_name):
                # print('{} is converted'.format(k))
                if k.replace(src_name, dst_name) in state_dict:
                    print('{} is duplicate'.format(k))
                state_dict[k.replace(src_name, dst_name)] = v
                converted = True
                break
        if not converted:
            state_dict[k] = v
            print('{} not converted'.format(k))

    for k in delete_keys:
        src_state_dict.pop(k)

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
