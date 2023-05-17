from torchvision import transforms
from .paws_helpers import Solarize, Equalize
from functools import partial
import torch
import random
import numpy as np
from .simclr_helpers import ConditionalResize
# ======================================================================
# These image transformations aren't the same as used in the torchvision repo, but they're good enough I think

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)

    color_distort = transforms.Compose([
        rnd_color_jitter,
        Solarize(p=0.2),
        Equalize(p=0.2)])
    return color_distort

def normalize():
    transform = transforms.Compose(
            [transforms.Normalize(
                 (0.4914, 0.4822, 0.4465),
                 (0.2023, 0.1994, 0.2010))])
    return transform

# ===================================================================
color_distortion = 0.5

def validation(norm=True):
    transform_val = transforms.Compose([
        ConditionalResize(32),
        transforms.CenterCrop(size=32),
        transforms.ToTensor(),
        normalize() if norm else lambda x:x 
    ])
    return transform_val


def training(norm=True):
    transform_train = transforms.Compose(
            [
             transforms.RandomResizedCrop(size=32, scale=(0.75, 1.0)),
             transforms.RandomHorizontalFlip(),
             get_color_distortion(s=color_distortion),
             transforms.ToTensor(),
             normalize() if norm else lambda x:x 
        ])
    return transform_train

def training_multicrop(norm=True):
    transform_train = transforms.Compose(
            [
             transforms.RandomResizedCrop(size=18, scale=(0.3, 0.75)),
             transforms.RandomHorizontalFlip(),
             get_color_distortion(s=color_distortion),
             transforms.ToTensor(),
             normalize() if norm else lambda x:x 
        ])
    return transform_train

