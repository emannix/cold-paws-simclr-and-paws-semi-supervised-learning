# https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/mnist-hello-world.html
# https://github.com/PyTorchLightning/pytorch-lightning/issues/2457

import torch
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torchmetrics.functional.classification.accuracy import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from src.utils import PredictionCache, expand_inputs, log_sum_exp
from src.utils import _ECELoss
from src.callbacks import RepresentationCache
from torch.distributions.normal import Normal
from torch.optim import lr_scheduler

from operator import attrgetter
from experiments.model_base_s import ModelBaseSupervised
import numpy as np

from pdb import set_trace as pb

import time
from argparse import Namespace
from torchvision import transforms
import copy
from torch import distributions as dist

class PyzFinetune(ModelBaseSupervised):
    def __init__(self, config):
        super(PyzFinetune, self).__init__(config)
        conf = Namespace(**self.config.experiment_params)
        # =======================================================================

        if hasattr(conf, 'model_path'):
            self.model_path = conf.model_path
            model = conf.model_class.load_from_checkpoint(conf.model_path)
        else:
            model = conf.model

        self.classes = conf.classes

        self.model = model
        if hasattr(self.model, 'set_head_projection'):
            self.model.set_head_projection(conf.use_head_proj)

        if self.save_representation_bool:
            for key, cache in self.representation_cache.items():
                if key != 'label' and key != 'index':
                    if key == 'before_global_pool':
                        try:
                            retriever = attrgetter('encoder.model.net')
                            retriever(self.model).register_forward_hook(cache.hook)
                        except:
                            retriever = attrgetter('encoder.net')
                            retriever(self.model).register_forward_hook(cache.hook)
                    else:
                        retriever = attrgetter(key)
                        retriever(self.model).register_forward_hook(cache.hook)
        
        self.freeze_encoder = conf.freeze_encoder
        if self.freeze_encoder == True:
            self.freeze_encoder = 1e6
        elif self.freeze_encoder == False:
            self.freeze_encoder = -1

        self.discriminator = conf.discriminator
        self.z_points = conf.z_points
        self.z_points_use = self.z_points

        self.loss_discr = conf.loss_discr
        self.last_layer = conf.last_layer

        self.ece_criterion = _ECELoss()

    def base_step(self, x_input_X, y_input_X, idx):
        if hasattr(self, 'model_path'):
            z_x, z_x_p = self.model(x_input_X, grad=self.freeze_encoder < self.current_epoch and self.model.training, gen=False)
        else:
            z_x = self.model(x_input_X)

        if self.save_representation_bool:
            if 'index' in self.representation_cache:
                self.representation_cache['index'].update(idx)
            if 'label' in self.representation_cache:
                self.representation_cache['label'].update(y_input_X)

        logit_py_z = self.discriminator(z_x)
        py_z = self.last_layer(logit_py_z)

        if (y_input_X is not None):
            log_py_z = self.loss_discr(y_input_X, logit_py_z)
        else:
            log_py_z = torch.tensor(0.0)

        if (self.z_points_use > 1):
            py_z = py_z.view(py_z.shape[0]//self.z_points_use, self.z_points_use, self.classes)
            z_x = z_x.view(z_x.shape[0]//self.z_points_use, self.z_points_use, z_x.shape[1])
            if (y_input_X is not None):
                log_py_z = log_py_z.view(log_py_z.shape[0]//self.z_points_use, self.z_points_use)

        return py_z, log_py_z, logit_py_z, z_x

# =============================================================================================

    def _full_step(self, py_z, log_py_z, y):

        if (self.z_points_use > 1):
            py_x = torch.mean(py_z, dim=1)
            log_py_x = torch.mean(log_py_z, dim=1)
        else:
            py_x = py_z
            log_py_x = log_py_z

        if hasattr(self, 'weight_classes'):
            class_weights = self.trainer.datamodule.class_weights.to(log_py_x)
            log_py_x = log_py_x*class_weights[y]

        total_loss = -log_py_x.mean()

        if not torch.isfinite(total_loss).all():
            pb()
        # =========================================================================================
        py_x = torch.repeat_interleave(py_x, self.z_points_use, dim=0)
        return_prob = {'acc': py_x, 'ece': self.ece_criterion(py_x, y)}
        return total_loss, return_prob

    def full_step(self, batch, total_batches):
        x_input_X, y_input_X, idx = batch
        py_z, log_py_z, logit_py_z, z_x = self.base_step(x_input_X, y_input_X, idx)
        total_loss, return_prob = self._full_step(py_z, log_py_z, y_input_X)

        return_prob['label'], return_prob['index'] = y_input_X, idx
        return total_loss, return_prob

# =============================================================================================
    def forward(self, x, y_input_X=None, idx=None):
        self.model.eval()
        self.discriminator.eval()
        with torch.no_grad():
            py_z, log_py_z, logit_py_z, z_x = self.base_step(x, y_input_X, idx)

            if hasattr(self, 'init_data_params'): # this is now deprecated really...
                if self.init_data_params.representation_output == 'projection_head':
                    z_x = self.model.encoder_mapping(z_x)

        return py_z, z_x

# =============================================================================================

    def predict_step(self, batch, batch_idx):
        x, y, idx = batch
        py_x, z_x = self.forward(x, y_input_X=y, idx=idx)

        if self.z_points > 1:
            idx = idx.view(idx.shape[0]//self.z_points, self.z_points)
            idx = idx[:, 0]

        return py_x.detach(), z_x.detach(), y, idx