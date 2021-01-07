r"""Runs Hyperpixel Flow framework"""

import argparse
import os
import os.path as osp
import logging
import time

from torch.utils.data import DataLoader
import torch

from model import hpflow, geometry, evaluation, util
from data import download
import mmcv
from mmcv import Config, DictAction


def run(datapath, benchmark, backbone, thres, alpha, hyperpixel,
        logfile, beamsearch, model=None, dataloader=None, visualize=False, ckpt_path=None):
    r"""Runs Hyperpixel Flow framework"""

    # 1. Logging initialization
    if not beamsearch:
        if visualize: os.mkdir(logfile + 'vis')

    # 2. Evaluation benchmark initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dataloader is None:
        download.download_dataset(os.path.abspath(datapath), benchmark)
        split = 'val' if beamsearch else 'test'
        dset = download.load_dataset(benchmark, datapath, thres, device, split)
        dataloader = DataLoader(dset, batch_size=1, num_workers=0)

    # 3. Model initialization
    if model is None:
        model = hpflow.HyperpixelFlow(backbone, hyperpixel, benchmark, device, ckpt_path=ckpt_path)
    else:
        model.hyperpixel_ids = util.parse_hyperpixel(hyperpixel)

    # 4. Evaluator initialization
    evaluator = evaluation.Evaluator(benchmark, device)

    for idx, data in enumerate(dataloader):

        # a) Retrieve images and adjust their sizes to avoid large numbers of hyperpixels
        data['src_img'], data['src_kps'], data['src_intratio'] = util.resize(data['src_img'], data['src_kps'][0])
        data['trg_img'], data['trg_kps'], data['trg_intratio'] = util.resize(data['trg_img'], data['trg_kps'][0])
        data['alpha'] = alpha

        # b) Feed a pair of images to Hyperpixel Flow model
        with torch.no_grad():
            confidence_ts, src_box, trg_box = model(data['src_img'], data['trg_img'])

        # c) Predict key-points & evaluate performance
        prd_kps = geometry.predict_kps(src_box, trg_box, data['src_kps'], confidence_ts)
        evaluator.evaluate(prd_kps, data)

        # d) Log results
        if not beamsearch:
            evaluator.log_result(idx, data=data)
        if visualize:
            vispath = os.path.join(logfile + 'vis', '%03d_%s_%s' % (idx, data['src_imname'][0], data['trg_imname'][0]))
            util.visualize_prediction(data['src_kps'].t().cpu(), prd_kps.t().cpu(),
                                      data['src_img'], data['trg_img'], vispath)
    if beamsearch:
        return (sum(evaluator.eval_buf['pck']) / len(evaluator.eval_buf['pck'])) * 100.
    else:
        evaluator.log_result(len(dset), data=None, average=True)


if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description='Hyperpixel Flow in pytorch')
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='pfpascal')
    parser.add_argument('--backbone', type=str)
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--hyperpixel', type=str, default='')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--auto-resume', action='store_true', help='automatically resume training')
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

    json_path = osp.join(cfg.work_dir, f'evaluate_seq_{args.dataset}.json')
    if osp.exists(json_path) and args.auto_resume:
        eval_info = mmcv.load(json_path)
        start_epoch = eval_info['last_epoch'] + 1
    else:
        eval_info = dict()
        start_epoch = args.start_epoch

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'evaluate_seq_{args.dataset}-{timestamp}.log')
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
        logging.info(f'evaluate converted {weight_path}')

        run(datapath=args.datapath, benchmark=args.dataset, backbone=args.backbone, thres=args.thres, alpha=args.alpha,
            hyperpixel=args.hyperpixel, logfile=log_file, beamsearch=False, visualize=args.visualize, ckpt_path=weight_path)
        if args.auto_resume:
            eval_info['last_epoch'] = epoch
            mmcv.dump(eval_info, json_path)
        if start_epoch == -1:
            break
