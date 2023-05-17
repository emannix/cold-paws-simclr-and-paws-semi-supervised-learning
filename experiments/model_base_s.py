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

# ===================================================================
        
class ModelBaseSupervised(ModelBase):
    def __init__(self, config):
        super(ModelBaseSupervised, self).__init__(config)

    def combined_step(self, batch_labelled, batch_idx, total_batches):
        start_time = time.time()
        total_loss_labelled, probs_dict = self.full_step(batch_labelled,
                total_batches
            )
        total_loss = total_loss_labelled

        # print("base_step --- %s seconds ---" % (time.time() - start_time))
        # print("epoch_step --- %s seconds ---" % (self.prev_start_time - start_time))
        self.prev_start_time = time.time()
        if len(batch_labelled) == 2:
            _, y = batch_labelled
        elif len(batch_labelled) == 3:
            _, y, i = batch_labelled
        elif len(batch_labelled) == 4:
            _, _, y, i = batch_labelled

        acc_dict = {}
        for key, value in probs_dict.items():
            if (len(value.shape) <= 1): #passing acc directly
                acc_dict[key] = value
            else:
                preds = torch.argmax(value, dim=1)
                acc = accuracy(preds, y, task='multiclass', num_classes=self.classes)
                acc_dict[key] = acc
        
        # print(total_loss)
        # print(acc)
        return total_loss, acc_dict, probs_dict, y

    def training_step(self, batch, batch_idx):

        total_loss, acc_dict, probs_dict, y = self.combined_step(batch['labelled'], 
                                    batch_idx, self.trainer.num_training_batches)

        self.log('train_loss', total_loss, prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        for key, value in acc_dict.items():
            if ('LongTensor' not in value.type()):
                self.log('train_'+key, value, prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
            if (hasattr(self, 'metric_probs_train')):
                if key in self.metric_probs_train:
                    if self.current_epoch+1 == self.trainer.max_epochs:
                        self.metric_cache_train[key](probs_dict[key], y)

        
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, acc_dict, probs_dict, y = self.combined_step(batch, 
                                                batch_idx, self.trainer.num_val_batches[0])

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', total_loss, prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        for key, value in acc_dict.items():
            if ('LongTensor' not in value.type()):
                self.log('val_'+key, value, prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
            self.metric_cache_val[key](probs_dict[key], y)
        return total_loss


    def test_step(self, batch, batch_idx):
        total_loss, acc_dict, probs_dict, y = self.combined_step(batch, 
                                                batch_idx, self.trainer.num_test_batches[0])
        self.log('test_loss', total_loss, prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        for key, value in acc_dict.items():
            if ('LongTensor' not in value.type()):
                self.log('test_'+key, value, prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
            self.metric_cache[key](probs_dict[key], y)
        return total_loss