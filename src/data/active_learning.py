import torch
import numpy as np
from pdb import set_trace as pb
from torch.utils.data import DataLoader, random_split, Subset
from .helpers import sample_balanced_by_class, extract_subset_dataset, calculate_class_weights
import pytorch_lightning as pl
import random

from ..data_samplers import *
from ..augmentation import augmentation_library

from torch.utils.data import ConcatDataset
import copy

class ActiveLearningDataTest(pl.LightningDataModule):
    def __init__(self, batch_size=100, 
            batch_size_unlabelled_scalar = 1,
            batch_size_validation_scalar = 1,
            dataset_labelled = None, 
            dataset_labelled_missing_size = 100,
            dataset_unlabelled = None,
            dataset_validation = None, 
            dataset_validation_missing_size = 100,
            dataset_test = None,
            workers=0,
            sampler_labelled=None,
            sampler_unlabelled=None,
            sampler_validation=None,
            sampler_conf=None,
            sampler_unlabelled_conf=None,
            sampler_val_conf=None,
            dual_loading=True,
            concat_loaders=False,
            sample_balanced=True,
            num_classes=10,
            unlabelled_transform = None,
            exclude_labelled = True,
            mask_unlabelled_targets = False,
            use_validation_transforms_for_predict = True,
            obj_data = 'inputs', obj_target = 'targets',
            predict_dataset = 'unlabelled',
            **kw
            ):
        super().__init__()
        self.batch_size = batch_size
        self.batch_size_unlabelled_scalar = batch_size_unlabelled_scalar
        self.batch_size_validation_scalar = batch_size_validation_scalar
        self.exclude_labelled = exclude_labelled
        self.use_validation_transforms_for_predict = use_validation_transforms_for_predict
        self.drop_last = True
        # ======================================================
        self.workers = workers
        
        self.mask_unlabelled_targets = mask_unlabelled_targets
        self.concat_loaders = concat_loaders
        self.dual_loading = dual_loading

        self.sample_balanced = sample_balanced
        self.num_classes = num_classes

        self.predict_dataset = predict_dataset

        self.pin_memory = True
        # ======================================================
        if dataset_labelled is None: # If dataset is not provided, sample from unlabelled data
            self.dataset_full = dataset_unlabelled
            self.unlabelled_mask = np.full((len(dataset_unlabelled),), True)
            self.labelled_mask = 1 - self.unlabelled_mask
        else: # If labelled dataset is provided, combine the two of them
              # Record which samples are labelled, and which are not
            self.dataset_full = torch.utils.data.ConcatDataset([dataset_labelled, dataset_unlabelled])
            self.dataset_full = extract_subset_dataset(self.dataset_full, obj_data=obj_data, obj_target=obj_target)
            labelled_mask = np.full((len(dataset_labelled),), True)
            unlabelled_mask = np.full((len(dataset_unlabelled),), False)
            self.labelled_mask = np.concatenate([labelled_mask, unlabelled_mask])
            self.unlabelled_mask = 1 - self.labelled_mask

        # ======================================================
        # Create unlabelled data as subset of full data
        self.dataset_unlabelled = Subset(self.dataset_full, np.nonzero(self.unlabelled_mask)[0])
        self.dataset_unlabelled = copy.deepcopy(self.dataset_unlabelled)
        if unlabelled_transform is not None:
            unlabelled_transform = augmentation_library(unlabelled_transform, None)
            self.dataset_unlabelled.dataset.transform = unlabelled_transform
        # ======================================================
        # Create labelled data as subset of full data
        self.dataset_labelled = Subset(self.dataset_full, np.nonzero(self.labelled_mask)[0])

        if dataset_labelled is None:
            # If labelled dataset is not provided, sample from unlabelled data
            if dataset_labelled_missing_size < len(self.dataset_labelled.dataset):
                subset_indices = self.extract_sample_indices_from_unlabelled(
                                          dataset_size=dataset_labelled_missing_size)
            else:
                subset_indices = np.arange(len(self.dataset_labelled.dataset))
            self.move_unlabelled_to_labelled(subset_indices)
            
        # ======================================================
        # if we have missing validation or training data, fill it from the unlabelled data
        if dataset_validation is None:
            subset_indices = self.extract_sample_indices_from_unlabelled(
                                          dataset_size=dataset_validation_missing_size)
            self.dataset_validation = self.extract_subset_from_unlabelled( subset_indices)
        else:
            self.dataset_validation = dataset_validation

        # If we are missing test data, use validation data
        if (dataset_test is None):
            self.dataset_test = self.dataset_validation
        else:
            self.dataset_test = dataset_test

        # ======================================================
        # Create sampler
        self.sampler_labelled = sampler_labelled
        self.sampler_unlabelled = sampler_unlabelled
        self.sampler_validation = sampler_validation

        self.sampler_conf = sampler_conf
        if sampler_unlabelled_conf is None:
            self.sampler_unlabelled_conf = sampler_conf
        else:
            self.sampler_unlabelled_conf = sampler_unlabelled_conf

        self.sampler_val_conf = sampler_val_conf
        if (sampler_labelled is not None):
            self.sampler_labelled = eval(sampler_labelled)
        if (sampler_unlabelled is not None):
            self.sampler_unlabelled = eval(sampler_unlabelled)
        if (sampler_validation is not None):
            self.sampler_validation = eval(sampler_validation)

        # ======================================================
        # Mask unlabelled data if necessary
        if mask_unlabelled_targets:
            self.dataset_unlabelled.dataset.targets = -torch.ones_like(self.dataset_unlabelled.dataset.targets)

        # Pre-create class weights
        self.class_weights = calculate_class_weights(self.dataset_labelled, self.num_classes)

        # self.dataset_labelled.indices
        # (self.dataset_labelled.dataset.targets[self.dataset_labelled.indices] == 0).sum()

    # ======================================================================
    def get_unlabelled_indices(self, subset_indices):
        indices = self.dataset_unlabelled.indices[subset_indices]
        return indices

    def get_labelled_indices(self, subset_indices):
        indices = self.dataset_labelled.indices[subset_indices]
        return indices

    def _update_indices(self):
        self.dataset_labelled.indices = np.nonzero(self.labelled_mask)[0]
        self.dataset_unlabelled.indices = np.nonzero(self.unlabelled_mask)[0]
    # ======================================================================

    def move_unlabelled_to_labelled(self, subset_indices, get_index=True):
        if (get_index):
            indices = self.get_unlabelled_indices(subset_indices)
        else:
            indices = subset_indices

        self.labelled_mask[indices] = True
        if self.exclude_labelled:
            self.unlabelled_mask[indices] = False
        self._update_indices()

    # def move_labelled_to_unlabelled(self, subset_indices):
    #     indices = self.get_labelled_indices(subset_indices)

    #     self.labelled_mask[indices] = False
    #     self.unlabelled_mask[indices] = True
    #     self._update_indices()

    def remove_from_unlabelled(self, subset_indices):
        indices = self.get_unlabelled_indices(subset_indices)
        self.unlabelled_mask[indices] = False
        self._update_indices()

    # ======================================================================
    def extract_sample_indices_from_unlabelled(self, dataset_size=100):
        if (self.sample_balanced):
            indices = sample_balanced_by_class(dataset = self.dataset_unlabelled, per_class = dataset_size//self.num_classes)
        else:
            indices = np.random.choice(np.arange(len(self.dataset_unlabelled)), size=dataset_size, replace=False)
        return indices

    def extract_subset_from_unlabelled(self, subset_indices):
        dataset_indices = self.get_unlabelled_indices(subset_indices)
        self.remove_from_unlabelled(subset_indices)
        return Subset(self.dataset_full, dataset_indices)

    # ======================================================================

    # def get_labelled_data(self):
    #   return extract_subset_dataset(self.dataset_labelled)

    # def get_unlabelled_data(self):
    #   return extract_subset_dataset(self.dataset_unlabelled)

    def get_unlabelled_targets(self):
        targets = torch.as_tensor(self.dataset_unlabelled.dataset.targets)
        return torch.as_tensor(targets)[self.dataset_unlabelled.indices]
    # ======================================================================

    def prepare_data(self):
        # build dataset
        pass

    def setup(self, stage=None):
        #
        pass

    def worker_init_fn(self, rank):
        pass
        # random.seed(0)

    def get_sampler(self, sampler, conf, dataset):
        conf.pop('batch_sampler', None)
        if (sampler is not None):
            sampler = sampler(dataset, batch_size=self.batch_size, **conf)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        return shuffle, sampler

    def get_loader(self, dataset, dataset_type='labelled'):
        if dataset_type == 'labelled':
            sampler_class = self.sampler_labelled
            sampler_conf = self.sampler_conf
            drop_last=self.drop_last
            batch_size=self.batch_size
        elif dataset_type == 'unlabelled':
            sampler_class = self.sampler_unlabelled
            sampler_conf = self.sampler_unlabelled_conf
            drop_last=self.drop_last
            batch_size=int(self.batch_size*self.batch_size_unlabelled_scalar)
        elif dataset_type == 'validation':
            sampler_class = self.sampler_validation
            sampler_conf = self.sampler_val_conf
            drop_last=False
            batch_size=int(self.batch_size*self.batch_size_validation_scalar)

        shuffle, sampler = self.get_sampler(sampler_class, copy.deepcopy(sampler_conf), dataset)
        
        if 'batch_sampler' in sampler_conf:
            if sampler_conf['batch_sampler']:
                return DataLoader(dataset = dataset, worker_init_fn=self.worker_init_fn,
                            num_workers=self.workers, batch_sampler=sampler, pin_memory=self.pin_memory)
        else:
            return DataLoader(dataset = dataset, worker_init_fn=self.worker_init_fn, 
                    batch_size=batch_size, shuffle=shuffle, 
                        num_workers=self.workers, drop_last=drop_last, sampler=sampler, pin_memory=self.pin_memory)


    # =============================================================================================================
    # https://github.com/Lightning-AI/lightning/issues/638
    # could refactor this code so that the dataloaders aren't recreated... but that's lots of effort
    def _train_dataloader(self):
        loaders = {}

        if (self.concat_loaders):
            concat_dataset = ConcatDataset([self.dataset_labelled, self.dataset_unlabelled])
            loaders = self.get_loader(concat_dataset, dataset_type='labelled')
        else:
            loaders['labelled'] = self.get_loader(self.dataset_labelled, dataset_type='labelled') 
            if (len(self.dataset_unlabelled) > 0 and self.dual_loading):
                loaders['unlabelled'] = self.get_loader(self.dataset_unlabelled, dataset_type='unlabelled')

        self.class_weights = calculate_class_weights(self.dataset_labelled, self.num_classes)
        return loaders

    def train_dataloader(self):
        if hasattr(self, 'loaders'):
            loaders = self.loaders
        else:
            loaders = self._train_dataloader()
            self.loaders = loaders
        return loaders
    # =============================================================================================================
    def _val_dataloader(self):
        val_loader =  self.get_loader(self.dataset_validation, dataset_type='validation')
        return val_loader

    def val_dataloader(self):
        if hasattr(self, 'val_loader'):
            val_loader = self.val_loader
        else:
            val_loader = self._val_dataloader()
            self.val_loader = val_loader
        return val_loader

    # =============================================================================================================
    def val_dataloader_swap(self):
        loaders = {}

        loaders['labelled'] = self.get_loader(self.dataset_validation, dataset_type='validation')
        self.val_loader = loaders
        return loaders

    # =============================================================================================================

    def test_dataloader(self):
        return self.get_loader(self.dataset_test, dataset_type='validation')

    # =============================================================================================================
    def _predict_dataloader(self):
        if self.predict_dataset == 'unlabelled':
            predict_dataset = self.dataset_unlabelled
        elif self.predict_dataset == 'labelled':
            predict_dataset = self.dataset_labelled

        predict_dataset = copy.deepcopy(predict_dataset)
        if self.use_validation_transforms_for_predict:
            try:
                predict_dataset.dataset.transform = self.dataset_validation.transform
            except:
                predict_dataset.dataset.transform = self.dataset_validation.dataset.transform
        loader = self.get_loader(predict_dataset, dataset_type='validation')
        loader.predict_dataset = copy.deepcopy(self.predict_dataset)
        loader.use_validation_transforms_for_predict = copy.deepcopy(self.use_validation_transforms_for_predict)
        return loader

    def predict_dataloader(self):
        if hasattr(self, 'pred_loader'):
            pred_loader = self.pred_loader
            if pred_loader.predict_dataset != self.predict_dataset or pred_loader.use_validation_transforms_for_predict != self.use_validation_transforms_for_predict:
                pred_loader = self._predict_dataloader()
                self.pred_loader = pred_loader
        else:
            pred_loader = self._predict_dataloader()
            self.pred_loader = pred_loader
        return pred_loader