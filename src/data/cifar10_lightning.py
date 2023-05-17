import torch
import numpy as np
from .active_learning import ActiveLearningDataTest
from .helpers import sample_balanced_by_class, extract_subset_dataset

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

import pytorch_lightning as pl

from PIL import Image
from pdb import set_trace as pb
from ..augmentation import augmentation_library
import copy

from . import cifar10_paws_lightning
from .helpers_data import parse_data_helper

class CIFAR10Index(datasets.CIFAR10):
    """Generate mini-batche pairs on CIFAR10 training set."""
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # .convert('RGB')
        img = self.transform(img)
        return img, target, idx  # stack a positive pair

class CIFAR10Dup(datasets.CIFAR10):
    """Generate mini-batche pairs on CIFAR10 training set."""
    def set_dedup_transform(self, transform):
        self.transform_dedup = transform

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # .convert('RGB')
        img_train = self.transform(img)
        img_dedup = self.transform_dedup(img)
        return img_dedup, img_train, target, idx  # stack a positive pair

def Cifar10DatasetModule(batch_size=100, labelled_samples=100, unlabelled_samples=1000, 
        validation_samples=0, exclude_labelled=True,
        dataset_wrapper = 'index', imbalance_ratio=1, data_seed=-1,
        transform_training='flatten', transform_training_custom=None, 
        transform_val='flatten', transform_val_custom=None,
        multicrop_transform_n=0, supervised_views=1, multicrop_transform=None,
        dedup_transform=None,
        **kw):

    transform_training = augmentation_library(transform_training, transform_training_custom)
    transform_val = augmentation_library(transform_val, transform_val_custom)
    if multicrop_transform is not None:
        multicrop_transform = augmentation_library(multicrop_transform, None)
    if dedup_transform is not None:
        dedup_transform = augmentation_library(dedup_transform, None)

    if (dataset_wrapper == 'index'):
        dataset = CIFAR10Index
    elif (dataset_wrapper == 'none'):
        dataset = datasets.CIFAR10
    elif (dataset_wrapper == 'paws'):
        dataset = CIFAR10Index
        dataset_paws = cifar10_paws_lightning.TransCIFAR10
    elif (dataset_wrapper == 'dedup'):
        dataset = CIFAR10Index
        dataset_dedup = CIFAR10Dup

    # elif (dataset_wrapper == 'unlabelled_transform'):
    #     dataset = CIFAR10DualTransform

    if data_seed != -1:
        st0 = np.random.get_state()
        np.random.seed(data_seed)

    if dataset_wrapper == 'imbalance':
        CIFAR10_train = dataset_imb(root="data", imbalance_ratio=imbalance_ratio, train=True, download=True, transform=transform_training)
    elif dataset_wrapper == 'paws':
        CIFAR10_train = dataset_paws(root="data", train=True, download=True, transform=transform_training, 
                multicrop_transform=(multicrop_transform_n, multicrop_transform), supervised_views=supervised_views)
    elif dataset_wrapper == 'dedup':
        CIFAR10_train = dataset_dedup(root="data", train=True, download=True, transform=transform_training)
    else:
        CIFAR10_train = dataset(root="data", train=True, download=True, transform=transform_training)
    test = dataset(root="data", train=False, transform=transform_val)

    # np.unique(CIFAR10_train.targets, return_counts=True)
    if (dataset_wrapper == 'dedup'):
        CIFAR10_train.set_dedup_transform(dedup_transform)

    training, validation, validation_missing_size = parse_data_helper(
                    CIFAR10_train, test, exclude_labelled, 
                    unlabelled_samples, labelled_samples, validation_samples, data_str='data', label_str = 'targets')

    loaders =  ActiveLearningDataTest(batch_size=batch_size, 
            dataset_labelled = None, 
            dataset_labelled_missing_size = labelled_samples,
            dataset_unlabelled = training, 
            dataset_validation = validation, 
            dataset_validation_missing_size = validation_missing_size,
            dataset_test = test,
            exclude_labelled = exclude_labelled, **kw)
    return loaders
