import torch
import numpy as np

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

import pytorch_lightning as pl

from pdb import set_trace as pb
from functools import partial

from . import simclr_cifar10 as simclr_cifar10
from . import paws_cifar10 as paws_cifar10
from . import paws_custom as paws_custom

# ==================================================

def augmentation_library(transform='flatten', transform_custom=None):
    transform_dictionary = {

        'simclr_cifar10_training_pt_normed': partial(simclr_cifar10.simclr, im_size = 32, color_dist = 0.5, blur=False, color_distort=False, normalize=True),
        'simclr_cifar10_simclr_pt_normed': partial(simclr_cifar10.simclr, im_size = 32, color_dist = 0.5, blur=False, color_distort=True, normalize=True),
        'simclr_cifar10_validation_normed': partial(simclr_cifar10.validation, im_size = 32, normalize=True),

        'simclr_big_cd0.5_training_pt_normed': partial(simclr_cifar10.simclr, seed=False, im_size = 224, color_dist = 0.5, blur=False, color_distort=False, normalize=True),
        'simclr_big_cd0.5_simclr_pt_normed': partial(simclr_cifar10.simclr, seed=False, im_size = 224, color_dist = 0.5, blur=False, color_distort=True, normalize=True),
        'simclr_imagenet_validation_pt_normed': partial(simclr_cifar10.validation, tensorflow=False, im_size = 224, normalize=True),

        'paws_cifar10_training': partial(paws_cifar10.training, norm=True),
        'paws_cifar10_training_multicrop': partial(paws_cifar10.training_multicrop, norm=True),
        'paws_cifar10_val': partial(paws_cifar10.validation, norm=True),

        'paws_custom_training': partial(paws_custom.training, norm=True),
        'paws_custom_training_multicrop': partial(paws_custom.training_multicrop, norm=True),
        'paws_custom_val': partial(paws_custom.validation, norm=True),
    }

    if (transform != 'custom'):
        transform = transform_dictionary[transform]() 
    elif (transform == 'custom'):
        transform = transform_custom
    return transform




