from torchvision import transforms
from .paws_helpers import Solarize, Equalize, GaussianBlur
from functools import partial
import torch
import random
import numpy as np
from pdb import set_trace as pb
# ======================================================================
# These image transformations aren't the same as used in the torchvision repo, but they're good enough I think

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    # rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        # rnd_gray,
        Solarize(p=0.2),
        Equalize(p=0.2)])
    return color_distort

def normalize():
    transform = transforms.Compose(
            [transforms.Normalize(
                 (0.485, 0.456, 0.406),
                 (0.229, 0.224, 0.225))])
    return transform

# ===================================================================
color_distortion = 0.5

def validation(norm=True):
    transform_val = transforms.Compose(
            [transforms.Resize(size=256),
             transforms.CenterCrop(size=224),
             transforms.ToTensor(),
             normalize() if norm else lambda x:x 
        ])
    return transform_val


def training(norm=True):
    transform_train = transforms.Compose(
                [
                 transforms.RandomResizedCrop(size=224, scale=(0.14, 1.0)),
                 transforms.RandomHorizontalFlip(),
                 get_color_distortion(s=color_distortion),
                 # GaussianBlur(p=0.5),
                 transforms.ToTensor(),
                 normalize() if norm else lambda x:x 
        ])
    return transform_train

def training_multicrop(norm=True, seed=False):
    transform_train = transforms.Compose(
            [
             transforms.RandomResizedCrop(size=96, scale=(0.05, 0.14)),
             transforms.RandomHorizontalFlip(),
             get_color_distortion(s=color_distortion),
             # GaussianBlur(p=0.5),
             transforms.ToTensor(),
             normalize() if norm else lambda x:x 
        ])
    return transform_train

