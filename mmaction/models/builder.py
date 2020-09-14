import torch.nn as nn
from mmcv.utils import build_from_cfg

from .registry import (BACKBONES, HEADS, LOCALIZERS, LOSSES, RECOGNIZERS,
                       WALKERS)


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, it is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_recognizer(cfg, train_cfg=None, test_cfg=None):
    """Build recognizer."""
    return build(cfg, RECOGNIZERS,
                 dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_walker(cfg, train_cfg=None, test_cfg=None):
    """Build walker."""
    return build(cfg, WALKERS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_localizer(cfg):
    """Build localizer."""
    return build(cfg, LOCALIZERS)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    args = cfg.copy()
    obj_type = args.pop('type')
    if obj_type in LOCALIZERS:
        return build_localizer(cfg)
    elif obj_type in RECOGNIZERS:
        return build_recognizer(cfg, train_cfg, test_cfg)
    elif obj_type in WALKERS:
        return build_walker(cfg, train_cfg, test_cfg)
    else:
        raise KeyError(f'{obj_type} not in any registry')
