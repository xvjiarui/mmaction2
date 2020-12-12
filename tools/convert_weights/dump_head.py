import argparse

import torch

dump_keys = ['head', 'loss']


def convert(src, dst):
    """Convert keys in detectron pretrained ResNet models to pytorch style."""
    # convert to pytorch style
    src_dict = torch.load(src)
    src_state_dict = src_dict.get('state_dict', src_dict)
    delete_keys = []
    for k, v in src_state_dict.items():
        for d_k in dump_keys:
            if d_k in k:
                delete_keys.append(k)
                continue

    for k in delete_keys:
        print(f'deleting {k}')
        src_state_dict.pop(k)

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = src_state_dict.copy()
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
