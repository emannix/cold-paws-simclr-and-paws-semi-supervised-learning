# https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/mnist-hello-world.html
# https://github.com/PyTorchLightning/pytorch-lightning/issues/2457

import torch
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torchmetrics.functional.classification.accuracy import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torch.distributions.normal import Normal

from torch.optim import lr_scheduler



from pdb import set_trace as pb

import time
from argparse import Namespace

from experiments.model_base import ModelBase

import gc

# ===================================================================

class ModelBaseSemiSupervised(ModelBase):
    def __init__(self, config):
        super(ModelBaseSemiSupervised, self).__init__(config)
        # self.labelled_batches = 0

    def combined_step(self, batch_labelled, batch_unlabelled, batch_idx, total_batches,
                        validate=False):
        # if (self.labelled_batches == 0):
        #     self.labelled_batches = len(self.trainer.datamodule.train_dataloader()['labelled'])
        #     self.unlabelled_batches = len(self.trainer.datamodule.train_dataloader()['unlabelled'])
        if (len(batch_labelled[0]) > 0):
            total_loss_labelled, probs_labelled = self.full_step(batch_labelled, 'labelled',
                    total_batches, validate=validate
                )
        else:
            total_loss_labelled, probs_labelled = torch.tensor(0.0), None

        if ((2-self.omega) > 0 and (batch_unlabelled is not None)): #  or batch_unlabelled[0].shape[0] > 0
            total_loss_unlabelled, probs_unlabelled = self.full_step(batch_unlabelled, 'unlabelled',
                    total_batches, validate=validate
                )
        else:
            total_loss_unlabelled, probs_unlabelled = torch.tensor(0.0), None

        # total_loss = total_loss_labelled*self.labelled_batches/total_batches\
        #                 +total_loss_unlabelled*self.unlabelled_batches/total_batches
        total_loss = total_loss_labelled*self.omega + total_loss_unlabelled*(2-self.omega)
        
        if len(batch_labelled) == 2:
            x, y = batch_labelled
        elif len(batch_labelled) == 3:
            x, y, i = batch_labelled
        elif len(batch_labelled) == 4:
            x1, x2, y, i = batch_labelled

        if (probs_labelled is None and probs_unlabelled is None):
            probs_dict = {}
        elif (probs_labelled is None):
            probs_dict = {**probs_unlabelled}
        elif (probs_unlabelled is None):
            probs_dict = {**probs_labelled}
        else:
            probs_dict = {**probs_labelled, **probs_unlabelled}

        acc_dict = {}
        if len(probs_dict) > 0:
            for key, value in probs_dict.items():
                if (len(value.shape) <= 1): #passing acc directly
                    acc_dict[key] = value
                else:
                    preds = torch.argmax(value, dim=1)
                    acc = accuracy(preds, y, task='multiclass', num_classes=self.classes)
                    acc_dict[key] = acc

        return total_loss, acc_dict, probs_dict, y


    def training_step(self, batch, batch_idx):
        if (self.config.data_params['concat_loaders']):
            newbatch = {}
            unlabelled_items = batch[1] == -1
            labelled_items = torch.where(~unlabelled_items)[0]
            unlabelled_items = torch.where(unlabelled_items)[0]

            if len(batch) == 2:
                newbatch['labelled'] = (torch.index_select(batch[0], 0,labelled_items), batch[1][labelled_items])
                newbatch['unlabelled'] = (torch.index_select(batch[0], 0,unlabelled_items), batch[1][unlabelled_items])
            else:
                newbatch['labelled'] = (torch.index_select(batch[0], 0,labelled_items), batch[1][labelled_items], batch[2][labelled_items])
                newbatch['unlabelled'] = (torch.index_select(batch[0], 0,unlabelled_items), batch[1][unlabelled_items], batch[2][unlabelled_items])
        else:
            newbatch = batch

        if 'unlabelled' not in newbatch:
            newbatch['unlabelled'] = None
        total_loss, acc_dict, probs_dict, y = self.combined_step(newbatch['labelled'], newbatch['unlabelled'], 
                                                batch_idx, self.trainer.num_training_batches*2)
                                                # As there is a labelled and unlabelled batch for each step
        # print("base_step --- %s seconds ---" % (time.time() - start_time))
        # print("epoch_step --- %s seconds ---" % (self.prev_start_time - start_time))
        # self.prev_start_time = time.time()

        self.log('train_loss', total_loss, prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        for key, value in acc_dict.items():
            if ('LongTensor' not in value.type()):
                self.log('train_'+key, value.detach().cpu(), prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
            if hasattr(self, 'metric_probs_train'):
                self.metric_cache_train[key](probs_dict[key], y)

        
        # self.end = time.time()
        # if hasattr(self, 'start'):
        #     print(self.end-self.start)
        #     pb()
        # self.start = time.time()

        return total_loss

    def validation_step(self, batch, batch_idx):

        total_loss, acc_dict, probs_dict, y = self.combined_step(batch, None, 
                                                batch_idx, self.trainer.num_val_batches[0]*2,
                                                validate=True)
        
        self.log('val_loss', total_loss, prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        for key, value in acc_dict.items():
            if ('LongTensor' not in value.type()):
                self.log('val_'+key, value.detach().cpu(), prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
            self.metric_cache_val[key](probs_dict[key], y)

        return total_loss


    def test_step(self, batch, batch_idx):
        total_loss, acc_dict, probs_dict, y = self.combined_step(batch, None, 
                                                batch_idx, self.trainer.num_test_batches[0]*2,
                                                validate=True)

        self.log('test_loss', total_loss, prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        for key, value in acc_dict.items():
            if ('LongTensor' not in value.type()):
                self.log('test_'+key, value.detach().cpu(), prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
            self.metric_cache[key](probs_dict[key], y)
        return total_loss

# ===================================================================

class ModelBaseSemiSupervisedCombined(ModelBaseSemiSupervised):
    def __init__(self, config):
        super(ModelBaseSemiSupervisedCombined, self).__init__(config)
        # self.labelled_batches = 0

    def combined_step(self, batch_labelled, batch_unlabelled, batch_idx, total_batches,
                        validate=False):
        # if (self.labelled_batches == 0):
        #     self.labelled_batches = len(self.trainer.datamodule.train_dataloader()['labelled'])
        #     self.unlabelled_batches = len(self.trainer.datamodule.train_dataloader()['unlabelled'])

        total_loss_labelled, probs_labelled, total_loss_unlabelled, probs_unlabelled = self.full_step(batch_labelled, batch_unlabelled,
                total_batches, validate=validate
            )

        # total_loss = total_loss_labelled*self.labelled_batches/total_batches\
        #                 +total_loss_unlabelled*self.unlabelled_batches/total_batches
        total_loss = total_loss_labelled*self.omega + total_loss_unlabelled*(2-self.omega)

        if len(batch_labelled) == 2:
            x, y = batch_labelled
        elif len(batch_labelled) == 3:
            x, y, i = batch_labelled
        elif len(batch_labelled) == 4:
            x1, x2, y, i = batch_labelled

        if (probs_labelled is None and probs_unlabelled is None):
            probs_dict = {}
        elif (probs_labelled is None):
            probs_dict = {**probs_unlabelled}
        elif (probs_unlabelled is None):
            probs_dict = {**probs_labelled}
        else:
            probs_dict = {**probs_labelled, **probs_unlabelled}

        acc_dict = {}
        if len(probs_dict) > 0:
            for key, value in probs_dict.items():
                if (len(value.shape) <= 1): #passing acc directly
                    acc_dict[key] = value
                else:
                    preds = torch.argmax(value, dim=1)
                    acc = accuracy(preds, y, task='multiclass', num_classes=self.classes)
                    acc_dict[key] = acc

        return total_loss, acc_dict, probs_dict, y
