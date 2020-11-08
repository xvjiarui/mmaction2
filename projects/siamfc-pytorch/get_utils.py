#  pip install py3nvml
import argparse
import time

import torch
import torch.nn as nn
from py3nvml.py3nvml import (nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                             nvmlDeviceGetMemoryInfo, nvmlDeviceGetName,
                             nvmlDeviceGetUtilizationRates, nvmlInit,
                             nvmlSystemGetDriverVersion)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('-f', '--freq', type=int, default=5, help='frequency')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.5, help='frequency')
    args = parser.parse_args()
    nvmlInit()
    print('Driver Version: {}'.format(nvmlSystemGetDriverVersion()))
    device_count = nvmlDeviceGetCount()
    handles = []
    fake_data_list = []
    conv_list = []
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        print('Device {}: {}'.format(i, nvmlDeviceGetName(handle)))
        mem_res = nvmlDeviceGetMemoryInfo(handle)
        mem_usage = mem_res.used / mem_res.total
        if mem_usage < 0.9:
            handles.append(handle)
            fake_data_list.append(
                torch.rand(32, 128, 56, 56, device=f'cuda:{i}'))
            conv_list.append(
                nn.Conv2d(128, 128, kernel_size=1).to(f'cuda:{i}'))
    while True:
        avg_gpu_utils = 0.
        for handle in handles:
            res = nvmlDeviceGetUtilizationRates(handle)
            avg_gpu_utils += res.gpu / 100
        avg_gpu_utils /= len(handles)
        print(f'utils: {avg_gpu_utils}')
        if avg_gpu_utils > args.threshold:
            print('sleeping')
            time.sleep(args.freq)
        else:
            start_time = time.time()
            while True:
                for data, conv in zip(fake_data_list, conv_list):
                    with torch.no_grad():
                        conv(data)
                if time.time() - start_time > args.freq:
                    break

        time.sleep(1)


if __name__ == '__main__':
    main()
