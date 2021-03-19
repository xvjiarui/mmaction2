import argparse
from collections import OrderedDict

import torch


def convert(src, dst):
    """Convert keys in detectron pretrained ResNet models to pytorch style."""
    # convert to pytorch style
    state_dict = OrderedDict()
    src_dict = torch.load(src)
    src_state_dict = src_dict.get('state_dict', src_dict)
    for k, v in src_state_dict.items():
        if not k.startswith("module.encoder_q."):
            continue
        old_k = k
        k = k.replace("module.encoder_q.", "")
        state_dict[k] = v
        print(f'{old_k} --> {k}')

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    checkpoint['meta'] = dict()
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src moc model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
