from experiments.model_simclr import SimCLR
from src.data import *
from src.utils import SpecifyModel
from src.functional import NTXent
from networks.simclrv2_resnet import SimCLRv2Resnet
from networks.simclrv2_resnet import get_resnet, get_head
from torch import nn
import numpy as np
from torchvision import transforms
import torch
from pdb import set_trace as pb

latent_dim = 512

encoder = SpecifyModel(
    get_resnet, {
        'depth':18, 'width_multiplier':1, 'sk_ratio':0.0, 'cifar_stem': True # 0.0625
    })

encoder_mapping = SpecifyModel(
    get_head, {
        'channels_in':latent_dim, 'num_layers':2, 'out_dim': 128
    })

loss_encoder = SpecifyModel(NTXent, {'temperature':0.2, 'distributed':True})


transform_training = 'simclr_cifar10_simclr_pt_normed'
transform_val = transform_training

from argparse import Namespace
# ===================================================
Config = Namespace(
# Trainer variables
    run_name = 'cifar10_resnet18sk0_simclr',
    version = 1,
    output_dir = 'output/',
    num_workers = 6,
    walltime = '0:59:59',
# ===================================================
    gpus = 1,
    strategy='myddp', 
    num_nodes=1,
    replace_sampler_ddp=False,
    sync_batchnorm=True,
# ===================================================
    benchmark = True,
    precision = 32,
# ===================================================
    max_epochs = 800,
    warm_up = 0.05,
    seed = 10,
    check_val_every_n_epoch = 10,
    enable_progress_bar = False,
    num_sanity_val_steps=2,
# ===================================================
    save_prediction_results=True,
    prediction_results_calc_AUC=False,
# ===================================================
# Metric monitoring
    metric_probs = ['overall_acc'],
    metric_monitor = 'val_overall_acc',
    metric_monitor_mode = 'max',
# ===================================================
# Data
    data_class = Cifar10DatasetModule,
    data_params = {
        'dual_loading': False,
        'sample_balanced': False,
        'sampler_labelled': 'RandomSamplerPair',
        'sampler_validation': 'RandomSamplerPair',
        'sampler_conf': {'seed': 10, 'shuffle': True},
        'sampler_val_conf': {'seed': 10, 'shuffle': False},
        'batch_size': 1024*2, # 1024*2,
        'labelled_samples':50000, 
        'unlabelled_samples':0,
        'transform_training': transform_training,
        'transform_training_custom':None,
        'transform_val':transform_val,
        'transform_val_custom':None,
        'data_includes': 'all',
    },
# ===================================================
# Experiment
    experiment_class = SimCLR,
    experiment_params = {
        'optimizer': 'mySGDLARS',
        'schedule': 'LinearWarmupCosineAnnealingLR',
        'momentum': 0.9,
        'momentum_nesterov': False,
        'momentum_version': 'tf',
        'parameter_groups_wd_exclude':['bias', 'bn'],
        'parameter_groups_lars_exclude':['bias', 'bn', 'discriminator'],
        'learning_rate':3.2,
        'min_learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'omega': 1.0,
        'encoder': encoder,
        'encoder_mapping':encoder_mapping,
        'loss_encoder': loss_encoder,
        'classes': 10
    },
# ===================================================
)

