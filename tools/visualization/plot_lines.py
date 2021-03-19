import argparse
import os.path as osp

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pgf import PdfPages

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
## for Palatino and other serif fonts use:
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'Times New Roman',
    # "font.serif": ["Palatino"],
    'font.size': 15,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
})


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 plot')
    parser.add_argument('data', help='input data')
    parser.add_argument('out', help='output file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = pd.read_csv(args.data, index_col=0)
    is_r50 = True
    if is_r50:
        layer3 = 6
        layer4 = 3
    else:
        layer3 = 2
        layer4 = 2
    # plot davis
    # data = data * 100
    # ax = data.plot(style=['-'] * layer3 + ['-.'] * layer4)
    # ax.legend([rf'res$_4$.b$_{i+1}$' for i in range(layer3)] + [rf'res$_5$.b$_{i+1}$' for i in range(layer4)])
    # # ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    # ax.yaxis.set_major_locator(ticker.FixedLocator([58, 62, 66]))
    # ax.yaxis.set_minor_locator(ticker.MaxNLocator(25))
    # ax.xaxis.set_major_locator(ticker.FixedLocator([0, 100]))
    # # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax.xaxis.set_minor_locator(ticker.MaxNLocator(10))
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # # ax.set_ylim([data.min().min(), data.max().max()])
    # ax.set_ylim([58, 66])
    # ax.set_xlim([0, 100])
    # ax.set_xlabel(xlabel='Epoch', labelpad=-5, fontsize=18)
    # ax.set_ylabel(r'$\mathcal{J\&F}_m$', labelpad=5, fontsize=18)
    # # plt.show()
    # plt.savefig(args.out, backend='pgf', bbox_inches = 'tight', pad_inches = 0)
    data = data * 100
    ax = data.plot(style=['-'] * layer3 + ['-.'] * layer4)
    ax.legend([rf'res$_4$.b$_{i+1}$' for i in range(layer3)] +
              [rf'res$_5$.b$_{i+1}$' for i in range(layer4)])
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.FixedLocator([60, 65, 70]))
    ax.yaxis.set_minor_locator(ticker.MaxNLocator(25))
    ax.xaxis.set_major_locator(ticker.FixedLocator([0, 500]))
    # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_ylim([data.min().min(), data.max().max()])
    ax.set_ylim([60, 70])
    ax.set_xlim([0, 500])
    ax.set_xlabel(xlabel='Epoch', labelpad=-5, fontsize=18)
    ax.set_ylabel(r'$\mathcal{J\&F}_m$', labelpad=5, fontsize=18)
    # plt.show()
    plt.savefig(args.out, backend='pgf', bbox_inches='tight', pad_inches=0)

    # ax = data.plot(style=['-'] * layer3 + ['-.'] * layer4)
    # # ax.legend([r'res$_4$.b1', r'res$_4$.b2', r'res$_5$.b1', r'res$_5$.b2'])
    # ax.legend([rf'res$_4$.b$_{i+1}$' for i in range(layer3)] + [rf'res$_5$.b$_{i+1}$' for i in range(layer4)])
    # # ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    # ax.yaxis.set_major_locator(ticker.FixedLocator([40, 50, 60]))
    # ax.yaxis.set_minor_locator(ticker.MaxNLocator(25))
    # ax.xaxis.set_major_locator(ticker.FixedLocator([0, 500]))
    # # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax.xaxis.set_minor_locator(ticker.MaxNLocator(50))
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # # ax.set_ylim([data.min().min(), data.max().max()])
    # ax.set_ylim([40, 60])
    # ax.set_xlim([0, 500])
    # ax.set_xlabel(xlabel='Epoch', labelpad=-5, fontsize=15)
    # ax.set_ylabel(ylabel='Precision', labelpad=5, fontsize=15)
    # plt.show()
    plt.savefig(args.out, backend='pgf')


if __name__ == '__main__':
    main()
