import os.path as osp
import time

import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from got10k.trackers import Tracker
from mmcv.runner import load_checkpoint
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from mmaction.models import build_model
from . import ops
from .datasets import Pair
from .heads import SiamConvFC
from .losses import BalancedLoss
from .transforms import SiamFCTransforms


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, z, x):
        z = self.backbone.extract_feat_test(z)
        x = self.backbone.extract_feat_test(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, cfg, logger):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = cfg
        self.logger = logger

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        model = build_model(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        # load checkpoint if provided
        if cfg.pretrained is not None:
            load_checkpoint(
                model, cfg.pretrained, map_location=self.device, logger=logger)
            logger.info(f'pretrained from {cfg.pretrained}')
        logger.info(f'Model: {str(model)}')
        if cfg.get('freeze_extractor', True):
            for param in model.parameters():
                param.requires_grad = False
        self.net = Net(
            backbone=model,
            head=SiamConvFC(
                cfg.model.cls_head.in_channels,
                128,
                out_scale=self.cfg.out_scale))
        if cfg.load_from is not None:
            load_checkpoint(
                self.net,
                cfg.load_from,
                map_location=self.device,
                logger=logger)
            logger.info(f'loaded from {cfg.load_from}')

        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

        # setup lr scheduler
        gamma = np.power(self.cfg.ultimate_lr / self.cfg.initial_lr,
                         1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2, box[0] - 1 +
            (box[2] - 1) / 2, box[3], box[2]
        ],
                       dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz), np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step**np.linspace(
            -(self.cfg.scale_num // 2), self.cfg.scale_num // 2,
            self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img,
            self.center,
            self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        # exemplar features
        z = torch.from_numpy(z).to(self.device).permute(
            2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone.extract_feat_test(z)

    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [
            ops.crop_and_resize(
                img,
                self.center,
                self.x_sz * f,
                out_size=self.cfg.instance_sz,
                border_value=self.avg_color) for f in self.scale_factors
        ]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).float()

        # responses
        x = self.net.backbone.extract_feat_test(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([
            cv2.resize(
                u, (self.upscale_sz, self.upscale_sz),
                interpolation=cv2.INTER_CUBIC) for u in responses
        ])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale = (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]
        ])

        return box

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        return boxes, times

    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)

            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs):
        # set to train mode
        self.net.train()

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(seqs=seqs, transforms=transforms)

        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                if (it + 1) % self.cfg.log_config.interval == 0:
                    self.logger.info('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                        epoch + 1, it + 1, len(dataloader), loss))
            # update lr at each epoch
            self.lr_scheduler.step()

            if epoch % (self.cfg.checkpoint_config.interval * 5) == 0:
                save_dir = osp.join(self.cfg.work_dir, 'siamfc')
                # save checkpoint
                mmcv.mkdir_or_exist(save_dir)
                net_path = osp.join(save_dir, f'epoch_{epoch + 1}.pth')
                torch.save(self.net.state_dict(), net_path)
                self.logger.info(f'{net_path} saved')
                dst_file = osp.join(save_dir, 'latest.pth')
                mmcv.symlink(osp.basename(net_path), dst_file)

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(
                dist <= r_pos, np.ones_like(x),
                np.where(dist < r_neg,
                         np.ones_like(x) * 0.5, np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()

        return self.labels
