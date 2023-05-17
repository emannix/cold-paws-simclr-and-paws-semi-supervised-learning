# https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/mnist-hello-world.html
# https://github.com/PyTorchLightning/pytorch-lightning/issues/2457

import torch
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torchmetrics.functional.classification.accuracy import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import math

from src.utils import PredictionCache
from src.callbacks import RepresentationCache
from torch.distributions.normal import Normal

from src.cold_start_queries import create_query

from src.utils import SpecifyModel
# from pl_bolts.optimizers import LARS
from src.optimizers import optimizer_library, set_parameter_groups
from src.schedulers import scheduler_library

from pdb import set_trace as pb

import time
from argparse import Namespace
from scipy import stats

class ModelBase(LightningModule):
    def __init__(self, config):
        super(ModelBase, self).__init__()
        # =======================================================================
        if isinstance(config, dict):
            config = Namespace(**config)
        self.config = config
        # =======================================================================
        config.networks = {}
        for key, value in config.experiment_params.items():
            if (isinstance(value, SpecifyModel)):
                config.networks[key] = value.stringify()
        if hasattr(self.config, 'test'):
            if self.config.test == False:
                self.save_hyperparameters(config)
        # =======================================================================
        for key, value in config.experiment_params.items():
            if (isinstance(value, SpecifyModel)):
                config.experiment_params[key] = value.init_model()
        # =======================================================================
        conf = Namespace(**config.experiment_params)

        self.learning_rate = conf.learning_rate
        self.weight_decay = conf.weight_decay
        if hasattr(conf, 'parameter_groups_wd_exclude'):
            self.parameter_groups_wd_exclude = conf.parameter_groups_wd_exclude

        self.momentum = conf.momentum
        if hasattr(conf, 'momentum_nesterov'):
            self.momentum_nesterov = conf.momentum_nesterov
        if hasattr(conf, 'momentum_version'):
            self.momentum_version = conf.momentum_version

        if hasattr(conf, 'min_learning_rate'):
            self.min_learning_rate = conf.min_learning_rate
        if hasattr(conf, 'start_learning_rate'):
            self.start_learning_rate = conf.start_learning_rate
        if hasattr(conf, 'final_learning_rate'):
            self.final_learning_rate = conf.final_learning_rate


        self.omega = conf.omega
        self.classes = conf.classes

        if hasattr(conf, 'weight_classes'):
            if conf.weight_classes:
                self.weight_classes = conf.weight_classes

        if hasattr(conf, 'learning_rate_model'):
            self.learning_rate_model = conf.learning_rate_model
        if hasattr(conf, 'learning_rate_decoderencoder'):
            self.learning_rate_decoderencoder = conf.learning_rate_decoderencoder

        if (hasattr(conf, 'save_representation')):
            self.representation_cache = {}
            for key in conf.save_representation:
                self.representation_cache[key] = RepresentationCache()
            self.save_representation_bool = True
            self.save_representation = conf.save_representation
        else:
            self.save_representation_bool = False
        # =======================================================================
        self.optimizer = conf.optimizer
        self.schedule = conf.schedule
        # =======================================================================        
        # pl.seed_everything(config.seed)

        self.prev_start_time = 0
        # =======================================================================
        self.metric_probs = config.metric_probs
        self.metric_cache = dict(zip(self.metric_probs, [PredictionCache() for x in self.metric_probs]))
        self.metric_cache_val = dict(zip(self.metric_probs, [PredictionCache() for x in self.metric_probs]))
        if (hasattr(config, 'metric_probs_train')):
            self.metric_probs_train = config.metric_probs_train
            self.metric_cache_train = dict(zip(self.metric_probs_train, [PredictionCache() for x in self.metric_probs_train]))
        # =======================================================================
        if ('selected_labels' in self.config.data_params):
            init_data_params = self.config.data_params['selected_labels']
            self.init_data_params = Namespace(**init_data_params)
        # =======================================================================

    # ======================================================================
    def setup_optimizers(self):
        if not hasattr(self, 'parameter_groups_wd_exclude'):
            if (self.optimizer == 'SGD-fixmatch'):
                self.parameter_groups_wd_exclude = ['bias', 'bn']
            if (self.optimizer == 'SGD-simclr'):
                self.parameter_groups_wd_exclude = ['bn']
            if (self.optimizer == 'LARS'):
                self.parameter_groups_wd_exclude = ['bias', 'bn', 'discriminator']
        if 'parameter_groups_lars_exclude' in self.config.experiment_params:
            self.parameter_groups_lars_exclude = self.config.experiment_params['parameter_groups_lars_exclude']
        else:
            self.parameter_groups_lars_exclude = self.parameter_groups_wd_exclude
        grouped_parameters = set_parameter_groups(self)

        optimizer = optimizer_library(self, grouped_parameters)
        scheduler = scheduler_library(self, optimizer)

        return [optimizer], [scheduler]

    # ======================================================================
    def setup_init_data(self):
        if (hasattr(self, 'init_data_params') and self.config.test == False): # and self.config.test == False
            datamodule = self.trainer.datamodule
            create_query(self.trainer, self, datamodule, self.init_data_params.policy, self.init_data_params.samples)

    # ======================================================================

    def configure_optimizers(self):
        self.setup_init_data()
        self.trainer.reset_train_dataloader(self)
        return self.setup_optimizers()

    # ======================================================================
    def reset_parameters(self, method):
        def conv2d_weight_truncated_normal_init(p):
            fan_in = p.shape[1]
            stddev = np.sqrt(1. / fan_in) / .87962566103423978
            r = stats.truncnorm.rvs(-2, 2, loc=0, scale=1., size=p.shape)
            r = stddev * r
            with torch.no_grad():
                p.copy_(torch.FloatTensor(r))

        def linear_normal_init(p):
            with torch.no_grad():
                # p.normal_(std=0.01)
                p.copy_(torch.FloatTensor(np.random.normal(size=p.shape)))

        if method == 'fixmatch':
            torch.manual_seed(0)
            np.random.seed(seed=233423)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight,
                                            mode='fan_out',
                                            nonlinearity='leaky_relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.Linear):
                    torch.manual_seed(0)
                    np.random.seed(seed=233423)
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
        elif method == 'simclrv1':
            for m in self.modules():
                torch.manual_seed(0)
                np.random.seed(seed=233423)
                if isinstance(m, nn.Conv2d):
                    conv2d_weight_truncated_normal_init(m.weight)
                elif isinstance(m, nn.Linear):
                    linear_normal_init(m.weight)
        elif method == 'simclrv2':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.ones_(m.weight)
                    # nn.init.constant_(m.weight, 0.01)
                    nn.init.dirac_(m.weight)
                    # pb()
                elif isinstance(m, nn.Linear):
                    # nn.init.ones_(m.weight)
                    # nn.init.constant_(m.weight, 0.01)
                    nn.init.eye_(m.weight)
        elif method == 'paws':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.weight, 0.01)
                    nn.init.constant_(m.bias, 0.00)
                elif isinstance(m, nn.Linear):
                    nn.init.eye_(m.weight)
                    nn.init.constant_(m.bias, 0.00)
