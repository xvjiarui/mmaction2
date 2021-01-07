r"""Beam search for hyperpixel layers"""

import argparse
import logging
import time
import os
import os.path as osp

from torch.utils.data import DataLoader
import torch

from data import dataset, download
from model import hpflow
from model import util
import evaluate
import mmcv
from mmcv import Config, DictAction


def parse_layers(layer_ids):
    r"""Parse list of layer ids (int) into string format"""
    layer_str = ''.join(list(map(lambda x: '%d,' % x, layer_ids)))[:-1]
    layer_str = '(' + layer_str + ')'
    return layer_str


def find_topk(membuf, kval):
    r"""Return top-k performance along with layer combinations"""
    membuf.sort(key=lambda x: x[0], reverse=True)
    return membuf[:kval]


def log_evaluation(layers, score, elapsed):
    r"""Log a single evaluation result"""
    logging.info('%20s: %4.2f %% %5.1f sec' % (layers, score, elapsed))


def log_selected(depth, membuf_topk):
    r"""Log selected layers at each depth"""
    logging.info(' ===================== Depth %d =====================' % depth)
    for score, layers in membuf_topk:
        logging.info('%20s: %4.2f %%' % (layers, score))
    logging.info(' ====================================================')


def beamsearch_hp(datapath, benchmark, backbone, thres, alpha, logpath,
                  candidate_base, candidate_layers, beamsize, maxdepth, ckpt_path=None):
    r"""Implementation of beam search for hyperpixel layers"""

    # 1. Model, and dataset initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = hpflow.HyperpixelFlow(backbone, '0', benchmark, device, ckpt_path)
    download.download_dataset(os.path.abspath(datapath), benchmark)
    dset = download.load_dataset(benchmark, datapath, thres, device, 'val')
    dataloader = DataLoader(dset, batch_size=1, num_workers=0)

    # 2. Search for the k-best base layers
    membuf_cand = []
    for base in candidate_base:
        start = time.time()
        hyperpixel = parse_layers(base)
        score = evaluate.run(datapath, benchmark, backbone, thres, alpha,
                             hyperpixel, logpath, True, model, dataloader)
        log_evaluation(base, score, time.time() - start)
        membuf_cand.append((score, base))
    membuf_topk = find_topk(membuf_cand, beamsize)
    score_sel, layer_sel = find_topk(membuf_cand, 1)[0]
    log_selected(0, membuf_topk)

    # 3. Proceed iterative search
    for depth in range(1, maxdepth):
        membuf_cand = []
        for _, test_layer in membuf_topk:
            for cand_layer in candidate_layers:
                if cand_layer not in test_layer and cand_layer > min(test_layer):
                    start = time.time()
                    test_layers = sorted(test_layer + [cand_layer])
                    if test_layers in list(map(lambda x: x[1], membuf_cand)):
                        break
                    hyperpixel = parse_layers(test_layers)
                    score = evaluate.run(datapath, benchmark, backbone, thres, alpha,
                                         hyperpixel, logpath, True, model, dataloader)

                    log_evaluation(test_layers, score, time.time() - start)
                    membuf_cand.append((score, test_layers))

        membuf_topk = find_topk(membuf_cand, beamsize)
        score_tmp, layer_tmp = find_topk(membuf_cand, 1)[0]

        if score_tmp > score_sel:
            layer_sel = layer_tmp
            score_sel = score_tmp
        log_selected(depth, membuf_topk)

    # 4. Log best layer combination and validation performance
    logging.info('\nBest layers, score: %s %5.3f' % (layer_sel, score_sel))

    return layer_sel


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Beam search for hyperpixel layers')
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='pfpascal')
    parser.add_argument('--backbone', type=str)
    parser.add_argument('--thres', type=str, default='bbox', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beamsize', type=int, default=4)
    parser.add_argument('--maxdepth', type=int, default=8)
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument( '--auto-resume', action='store_true', help='automatically resume training')
    parser.add_argument(
        '--start-epoch', type=int, default=1, help='-1 for latest.pth')
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--skip', action='store_true')
    args = parser.parse_args()

    if args.backbone is None:
        if '_r18_' in args.config:
            args.backbone = 'resnet18'
        elif '_r50_' in args.config:
            args.backbone = 'resnet50'
        elif '_r101_' in args.config:
            args.backbone = 'resnet101'
        else:
            raise ValueError('Invalid backbone')
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # work_dir is determined in this priority:
    # CLI > config file > default (base filename)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    json_path = osp.join(cfg.work_dir, f'beamsearch_seq_{args.dataset}.json')
    if osp.exists(json_path) and args.auto_resume:
        eval_info = mmcv.load(json_path)
        start_epoch = eval_info['last_epoch'] + 1
    else:
        eval_info = dict()
        start_epoch = args.start_epoch

    # 1. Candidate layers for hyperpixel initialization
    n_layers = {'resnet18': 8, 'resnet50': 17, 'resnet101': 34}
    candidate_base = [[i] for i in range(args.beamsize)]
    candidate_layers = list(range(n_layers[args.backbone]))

    # 2. Logging initialization
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'beamsearch_seq_{args.dataset}-{timestamp}.log')
    util.init_logger(log_file)
    util.log_args(args)

    for epoch in range(start_epoch, cfg.total_epochs + 1,
                       cfg.checkpoint_config.interval):
        if start_epoch == -1:
            args.auto_resume = False
            ckpt_path = osp.realpath(osp.join(cfg.work_dir, 'latest.pth'))
            epoch = int(
                osp.splitext(osp.basename(ckpt_path))[0].split('_')[-1])
        else:
            if epoch % args.eval_interval != 0:
                continue
            ckpt_path = osp.join(cfg.work_dir, f'epoch_{epoch}.pth')
            if not osp.exists(ckpt_path):
                if args.skip:
                    logging.info(f'{ckpt_path} not exist, skipping')
                    continue
                else:
                    logging.info(f'{ckpt_path} not exist, waiting')
            while not osp.exists(ckpt_path):
                time.sleep(300)
        logging.info(f'Found {ckpt_path}')

        weight_path = osp.realpath(ckpt_path).replace(
            'epoch_',
            osp.basename(args.config) + '_ep').replace('.pth', '-backbone.pth')
        if not osp.exists(weight_path):
            os.system(f'MKL_THREADING_LAYER=GNU python '
                      f'tools/convert_weights/convert_to_pretrained.py '
                      f'{ckpt_path} {weight_path}')
        logging.info(f'beamsearch converted {weight_path}')

        # 3. Run beam search
        logging.info('Beam search on \'%s validation split\' with \'%s\' backbone...\n' %
                     (args.dataset, args.backbone))
        layer_sel = beamsearch_hp(args.datapath, args.dataset, args.backbone, args.thres, args.alpha,
                                  log_file, candidate_base, candidate_layers, args.beamsize, args.maxdepth,
                                  ckpt_path=weight_path)
        if args.auto_resume:
            eval_info['last_epoch'] = epoch
            mmcv.dump(eval_info, json_path)
        if start_epoch == -1:
            break
