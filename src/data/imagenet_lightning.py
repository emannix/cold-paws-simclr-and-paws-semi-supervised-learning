import torch
import numpy as np
from .active_learning import ActiveLearningDataTest
from .helpers import sample_balanced_by_class, extract_subset_dataset
from .helpers_data import parse_data_helper

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

import pytorch_lightning as pl

from PIL import Image
from pdb import set_trace as pb
from ..augmentation import augmentation_library
import copy
import pandas as pd
import os

from .base_image_folder_dataset import ImageFolderCSVDataset, ImageFolder, TransImageDataset
from functools import partial

def ImagenetDatasetModule(batch_size=100, labelled_samples=100, unlabelled_samples=1000, 
        validation_samples=0, exclude_labelled=True,
        dataset_wrapper = 'index',
        data_includes = 'all',
        transform_training='flatten', transform_training_custom=None, 
        multicrop_transform_n=0, supervised_views=1, multicrop_transform=None,
        transform_val='flatten', transform_val_custom=None, **kw):

    transform_training = augmentation_library(transform_training, transform_training_custom)
    transform_val = augmentation_library(transform_val, transform_val_custom)
    if multicrop_transform is not None:
        multicrop_transform = augmentation_library(multicrop_transform, None)

    if (dataset_wrapper == 'index'):
        dataset = partial(ImageFolder, return_index=True)
        return_index = True
    elif (dataset_wrapper == 'none'):
        dataset = partial(ImageFolder, return_index=False)
        return_index = False
    elif (dataset_wrapper == 'paws'):
        dataset = partial(ImageFolder, return_index=True)
        return_index = True

    train_csv_file = 'imagenet-train.csv'
    train_folder = '/ILSVRC/Data/CLS-LOC/train/'
    test_csv_file = 'imagenet-val.csv'
    test_folder = '/ILSVRC/Data/CLS-LOC/val/'

    if data_includes in ['all']:
        data_subdir = ''
        data_path = '/imagenet/'
    elif data_includes == 'small':
        data_subdir = ''
        data_path = '/imagenet_small/'
    elif data_includes == 'imagenette':
        data_subdir = '/imagenet_subsets'
        data_path = '/imagenette2/'
        train_folder = 'train/'
        test_folder = 'val/'
    elif data_includes == 'imagewoof':
        data_subdir = '/imagenet_subsets'
        data_path = '/imagewoof2/'
        train_folder = 'train/'
        test_folder = 'val/'

    if 'PBS_JOBFS' in os.environ and os.environ['PBS_JOBFS'] != "":
        data_path = os.environ['PBS_JOBFS'] + data_path
    else:
        new_data_path = 'XXX/' + data_subdir + data_path
        if os.path.exists(new_data_path):
            data_path = new_data_path
        else:
            data_path = 'XXX/' + data_subdir + data_path

    train = dataset(root=data_path+train_folder, 
        transform=transform_training,
        # metadata_csv_file=data_path+train_csv_file, 
        # metadata_csv_image = 'image_path', 
        # metadata_csv_label = 'folder', image_ext = '',
        )
    if (dataset_wrapper == 'paws'):
        train = TransImageDataset(train, 
            multicrop_transform=(multicrop_transform_n, multicrop_transform), supervised_views=supervised_views)

    print('loaded train')
    if data_includes == 'imagenette' or data_includes == 'imagewoof':
        test = dataset(root=data_path+test_folder, 
                transform=transform_val,
                )
    else:
        test = ImageFolderCSVDataset(image_folder=data_path+test_folder, 
            transform=transform_val, return_index = return_index,
            metadata_csv_file=data_path+test_csv_file, 
            metadata_csv_image = 'image_path', 
            metadata_csv_label = 'folder', image_ext = '',
            )
    print('loaded test')

    # if data_includes in ['all', 'SCAN-50', 'SCAN-100', 'SCAN-200']:


    training, validation, validation_missing_size = parse_data_helper(
                    train, test, exclude_labelled, 
                    unlabelled_samples, labelled_samples, validation_samples)

    loaders =  ActiveLearningDataTest(batch_size=batch_size, 
            dataset_labelled = None, 
            dataset_labelled_missing_size = labelled_samples,
            dataset_unlabelled = training, 
            dataset_validation = validation, 
            dataset_validation_missing_size = validation_missing_size,
            dataset_test = None,
            exclude_labelled = exclude_labelled, **kw)
    print('created dataset')
    return loaders
