
from experiments.model_pyz_finetune import PyzFinetune

from experiments.model_simclr import SimCLR

from src.data import *
from src.utils import SpecifyModel
import PIL

from src.functional import LossLogCategorical
from networks.simclrv2_resnet import supervised_head
from torch import nn
import numpy as np
from torchvision import transforms
import torch
from pdb import set_trace as pb

# https://github.com/AndrewAtanov/simclr-pytorch/blob/master/configs/imagenet_eval.yaml

classes = 10
run_name = 'cifar10_supervised_finetune'
metric_monitor = 'val_acc'
data_class = Cifar10DatasetModule
class_transform = 'NA'
data_includes = 'all'
model_path = 'pretrained_models/cifar10_simclr.ckpt'
# =====================

latent_dim = 512

model_class = SimCLR
discriminator_z = SpecifyModel(
    supervised_head, {
        'input_dim':latent_dim, 'output_dim':classes, 'zero_init_logits_layer': True
    })

loss_discr = SpecifyModel(LossLogCategorical, {})
last_layer = SpecifyModel(nn.Softmax, {'dim':1})

transform_val = 'simclr_cifar10_validation_normed' 
transform_training = 'simclr_cifar10_training_pt_normed'

from argparse import Namespace
# ===================================================
Config = Namespace(
# Trainer variables
    run_name = run_name,
    version = 1,
    output_dir = 'output/',
    num_workers = 8,
    walltime = '0:59:59',
# ===================================================
    gpus = 1,
    strategy='myddp', 
    num_nodes=1,
    replace_sampler_ddp=False,
    sync_batchnorm=True,
# ===================================================
    benchmark = True,
    precision = 16,
    num_sanity_val_steps=2,
# ===================================================
    max_epochs = 100,
    early_stopping_patience = 20,
    seed = 10,
    check_val_every_n_epoch = 1,
    progress_bar_refresh_rate=20,
# ===================================================
    save_prediction_results=True,
    prediction_results_calc_AUC=True,
# ===================================================
# Metric monitoring
    metric_probs = ['acc', 'index', 'label', 'ece'],
    metric_monitor = metric_monitor,
    metric_monitor_mode = 'max',
# ===================================================
# Data
    data_class = data_class,
    data_params = {
        'num_classes': classes,
        'batch_size': 1024, 
        'sample_balanced':True,
        'dual_loading': False,
        'exclude_labelled': False,
        'sampler_labelled': 'MyDistributedSampler',
        'sampler_conf': {'shuffle': True, 'seed': 10},
        'sampler_validation': 'MyDistributedSampler',
        'sampler_val_conf': {'shuffle': False, 'seed': 10, 'drop_last': False},
        'labelled_samples':50000, 
        'unlabelled_samples':0,
        'validation_samples': 0,
        'transform_training': transform_training,
        'transform_val':transform_val,
        'class_transform': class_transform,
        'data_includes': data_includes,
    },
# ===================================================
# Experiment
    experiment_class = PyzFinetune,
    experiment_params = {
        'optimizer': 'mySGDLARS',
        'parameter_groups_wd_exclude':["bias", "bn"],
        'parameter_groups_lars_exclude':["bias", "bn", "discriminator"],
        'momentum_nesterov':False,
        'momentum_version':'tf',
        'schedule': 'Constant',
        'learning_rate':0.16,
        'min_learning_rate': 0,
        'weight_decay': 0,
        'momentum': 0.9,
        'omega': 1.0,
        'classes': classes,
        'z_points': 1,
        'freeze_encoder': False,
        'weight_classes': True,
        'use_head_proj': 0,
        'model_path': model_path,
        'model_class': model_class,
        'discriminator': discriminator_z,
        'loss_discr': loss_discr,
        'last_layer': last_layer,
    },
# ===================================================
)

