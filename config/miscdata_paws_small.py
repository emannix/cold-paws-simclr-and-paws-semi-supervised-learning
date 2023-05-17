from experiments.model_paws import PAWS
from experiments.model_simclr import SimCLR

from src.data import *
from src.utils import SpecifyModel
from src.functional import LossLogCategorical, LossLogBernoulli

from networks.paws_encoders import ProjectionHead

from torch import nn
import numpy as np
from torchvision import transforms
import torch
from pdb import set_trace as pb

classes = 10
latent_dim = 512
output_dim = 128

projection_head = SpecifyModel(
    ProjectionHead, {
        'hidden_dim':latent_dim, 'output_dim':output_dim
    })

transform_val = 'paws_cifar10_val' 
transform_training = 'paws_cifar10_training'
transform_training_multicrop = 'paws_cifar10_training_multicrop'

from argparse import Namespace
# ===================================================
# Important parameters
Config = Namespace(
# Trainer variables
    run_name = 'cifar10_paws_small',
    version = 1,
    output_dir = 'output/',
    num_workers = 6,
    gpus = 1,
    walltime = '0:59:59',
# ===================================================
    benchmark = True,
    precision = 16,
    num_sanity_val_steps=2,
# ===================================================
    strategy= 'myddp',#'ddp', 
    num_nodes=1,
    replace_sampler_ddp=False,
    sync_batchnorm=True,
# ===================================================
    max_epochs = 200,
    warm_up = 10,
    seed = 1,
    check_val_every_n_epoch = 10,
    progress_bar_refresh_rate=10,
    multiple_trainloader_mode='min_size',
# ===================================================
# ===================================================
# Metric monitoring
    metric_probs = ['acc', 'index', 'label'],
    metric_monitor = 'val_acc',
    metric_monitor_mode = 'max',
# ===================================================
# Data
    data_class = Cifar10DatasetModule,
    data_params = {
        'dataset_wrapper': 'paws',
        'supervised_views': 2,
        'multicrop_transform_n': 6,
        'multicrop_transform': transform_training_multicrop,
        'batch_size': 160,
        'batch_size_unlabelled_scalar': 1.6,
        'batch_size_validation_scalar': 1.6,
        'sampler_labelled': 'ClassStratifiedSampler',
        'sampler_conf': {'seed': 10, 'unique_classes':False, 'classes_per_batch':10, 'batch_sampler': True},
        'sampler_unlabelled': 'MyDistributedSampler',
        'sampler_unlabelled_conf': {'shuffle': True, 'seed': 10},
        'sampler_validation': 'MyDistributedSampler',
        'sampler_val_conf': {'shuffle': False, 'seed': 10},
        'dual_loading': True,
        'concat_loaders': False,
        'exclude_labelled': False,
        'labelled_samples':0,
        'unlabelled_samples':50000,
        'transform_training': transform_training,
        'transform_val':transform_val,
        'selected_labels': {
            'samples': 40,
            'policy': 'load_indices_vector',
            'load_indices_csv': 'indices/cifar10_finetune_bph_euclidean.csv' # 'indices/cifar10_random_labelled.csv'
        }
    },
# ===================================================
# Experiment
    experiment_class = PAWS,
    experiment_params = {
        'optimizer': 'pawsSGDLARS',
        'schedule': 'paws_WarmupCosineSchedule',
        'learning_rate':3.2,
        'start_learning_rate': 0.8,
        'final_learning_rate': 0.032,
        'weight_decay': 1.0e-04,
        'momentum': 0.9,
        'momentum_nesterov': False,
        'omega': 1.0,
        'classes': 10,
        'freeze_encoder': False,
        'model': None,
        'model_path': 'pretrained_models/cifar10_simclr.ckpt',
        'model_class': SimCLR,
        'use_head_proj': 0,
        'projection_head': projection_head,
        'label_smoothing': 0.1,
        'sharpen': 0.25,
        'me_max': True,
        'temperature': 0.1,
        'parameter_groups_wd_exclude': ['bias', 'bn'],
        'parameter_groups_lars_exclude': ['bias', 'bn']
    },
# ===================================================
)
