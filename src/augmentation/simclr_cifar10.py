
from torchvision import transforms
from .simclr_helpers import CenterCropAndResize, Clip, GaussianBlur, get_color_distortion, ConditionalResize
from functools import partial
import torch
import numpy as np
import random

from pdb import set_trace as pb
# ======================================================================
# These image transformations aren't the same as used in the torchvision repo, but they're good enough I think
# im_size = 32
# color_dist = 0.5

def normalize_cifar10():
    transform = transforms.Compose(
            [transforms.Normalize(
                 (0.4914, 0.4822, 0.4465),
                 (0.2023, 0.1994, 0.2010))])
    return transform

def normalize_imagenet():
    transform = transforms.Compose(
            [transforms.Normalize(
                 (0.485, 0.456, 0.406),
                 (0.229, 0.224, 0.225))])
    return transform
# ======================================================================

def validation(im_size = 32, normalize=False):
    if im_size == 32:
        transform_val = transforms.Compose(
            [
                ConditionalResize(im_size),
                transforms.ToTensor(),
                normalize_cifar10() if (normalize) else lambda x : x,
            ]
        )
    else:
        transform_val = transforms.Compose(
            [
                CenterCropAndResize(0.875, im_size),
                transforms.ToTensor(),
                normalize_imagenet() if (normalize) else lambda x : x,
            ]
        )
    return transform_val


def simclr(seed=False, im_size = 32, color_dist = 0.5, 
        blur=False, color_distort=False, normalize=False, rotation=False):

    transform_train = transforms.Compose([
        transforms.RandomRotation(45, interpolation=transforms.InterpolationMode.BICUBIC) if rotation else lambda x : x,
        transforms.RandomResizedCrop(
            im_size,
            scale=(0.08, 1.0), # seems like min of 10% of image can be cropped
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        get_color_distortion(s=color_dist, source='pytorch')  if color_distort else lambda x : x, # albumentations, pytorch, tf
        transforms.ToTensor(),
        GaussianBlur(im_size // 10, 0.5) if blur else lambda x : x,
        normalize_cifar10() if (normalize and im_size == 32) else lambda x : x,
        normalize_imagenet() if (normalize and im_size != 32) else lambda x : x,
        Clip() if (blur and not normalize) else lambda x : x
    ])
    return transform_train

