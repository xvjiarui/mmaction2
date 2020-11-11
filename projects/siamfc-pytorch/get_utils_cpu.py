import argparse
import time

import psutil
import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('-f', '--freq', type=int, default=5, help='frequency')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.5, help='threshold')
    parser.add_argument(
        '--memory', '-m', action='store_true', help='use memory')
    args = parser.parse_args()
    avg_cpu_utils = psutil.cpu_percent() / 100.
    fake_data = torch.rand(32, 128, 56, 56)
    conv = nn.Conv2d(128, 128, kernel_size=1)
    outputs = []
    while True:
        print(f'utils: {avg_cpu_utils}')
        if avg_cpu_utils > args.threshold:
            print('sleeping')
            time.sleep(args.freq)
        else:
            start_time = time.time()
            while True:
                with torch.no_grad():
                    conv_out = conv(fake_data)
                    if args.memory:
                        outputs.append(conv_out)
                        if len(outputs) > 20:
                            outputs.pop()
                if time.time() - start_time > args.freq:
                    break

        time.sleep(1)


if __name__ == '__main__':
    main()
