# https://github.com/p3i0t/SimCLR-CIFAR10

import torch
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torchmetrics.functional.classification.accuracy import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from src.utils import PredictionCache, expand_inputs, log_sum_exp
from torch.distributions.normal import Normal
from torch.optim import lr_scheduler

from experiments.model_base_s import ModelBaseSupervised
import numpy as np

from pdb import set_trace as pb

import time
from argparse import Namespace
from torchvision import transforms
import torchvision
import os
import copy
from torch import distributions as dist

class SimCLR(ModelBaseSupervised):
    def __init__(self, config):
        super(SimCLR, self).__init__(config)
        conf = Namespace(**self.config.experiment_params)
        # =======================================================================
        self.encoder = conf.encoder
        self.encoder_mapping = conf.encoder_mapping

        self.loss_encoder = conf.loss_encoder

        self.head_proj = 0

    def set_head_projection(self, use_head_proj):
        if use_head_proj != 0:
            self.head_proj = use_head_proj

            self.encoder_mapping_head = copy.deepcopy(self.encoder_mapping)
            self.encoder_mapping_head.model.layers = self.encoder_mapping.model.layers[0:use_head_proj]


# =============================================================================================

    def _forward(self, x):
        z_loc = self.encoder(x)
        z_mapping = self.encoder_mapping(z_loc)
        if self.head_proj > 0:
            z_loc = self.encoder_mapping_head(z_loc)
        return z_loc, z_mapping

    def forward(self, x, grad=None, **kw):
        if grad is None:
            return self._forward(x)
        else:
            if not grad:
                self.encoder.eval()
                self.encoder_mapping.eval()
                if self.head_proj > 0:
                    self.encoder_mapping_head.eval()

            with torch.set_grad_enabled(grad):
                return self._forward(x)

# =============================================================================================

    def full_step(self, batch, total_batches):
        reweight=False
        dedup_method = None
        embeddings = None

        x, y, _ = batch

        z_loc, z_mapping = self.forward(x)

        log_encoder_prob, acc = self.loss_encoder(z_mapping, y)

        # ================================================================
        total_loss = -log_encoder_prob.mean()
                
        if not torch.isfinite(total_loss).all():
            pb()
        # =========================================================================================
        return_prob = {'overall_acc': acc.mean()}
        return total_loss, return_prob

# =============================================================================================
