#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator
from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
import mmcv
import wandb
import json

@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    As described in the MOCO paper, there is an extra BN layer
    following the res5 stage.
    """
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if "coco" in dataset_name:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert "voc" in dataset_name
            return PascalVOCDetectionEvaluator(dataset_name)


def setup(args):
    cfg = get_cfg()
    cfg.MMACTION_CFG = ""
    cfg.PRETRAINED_CKPT = ""
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = get_cfg()
    cfg.MMACTION_CFG = ""
    cfg.PRETRAINED_CKPT = ""
    cfg.merge_from_list(args.opts)
    pretrained_ckpt = cfg.PRETRAINED_CKPT
    if len(pretrained_ckpt):
        assert os.path.exists(pretrained_ckpt)
        weight_path = os.path.realpath(pretrained_ckpt).replace('epoch_', os.path.basename(cfg.MMACTION_CFG)+'_ep').replace('.pth', '-d2.pkl')
        os.system(f'MKL_THREADING_LAYER=GNU python projects/moco/detection/convert-mmaction-to-detectron2.py {pretrained_ckpt} {weight_path}')
        args.opts.extend(['MODEL.WEIGHTS', weight_path])
    # if len(cfg.MMACTION_CFG) and len(pretrained_ckpt):
    #     mmaction_cfg = mmcv.Config.fromfile(cfg.MMACTION_CFG)
    #     for h in mmaction_cfg.log_config.hooks:
    #         if h.type == 'WandbLoggerHook':
    #             wandb_cfg = h.init_kwargs.to_dict()
    #             os.makedirs(f'wandb/{os.path.basename(weight_path)}', exist_ok=True)
    #             wandb_cfg.update(
    #                 dict(
    #                     name=os.path.basename(weight_path),
    #                     resume=False,
    #                     dir=f'wandb/{os.path.basename(weight_path)}',
    #                     tags=[wandb_cfg['tags'], 'detectron2']))
    #             wandb.init(**wandb_cfg)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    # if len(cfg.MMACTION_CFG):
    #     with open('output/metrics.json', 'r') as f:
    #         for line in f:
    #             log = json.loads(line)
    #             iterations = log.pop('iteration')
    #             wandb.log(log, step=iterations)
