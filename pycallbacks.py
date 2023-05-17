from pytorch_lightning.callbacks import Callback
from pdb import set_trace as pb

from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
import os
import pandas as pd
import glob

import torch
import numpy as np
import torch.nn as nn

from torchmetrics import AUROC, AveragePrecision
# from torchmetrics.functional import average_precision
from torchmetrics import Metric
# from sklearn.metrics import average_precision_score, roc_auc_score

import torch.distributed as dist


class SaveResults(Callback):

    def __init__(self, config):
        self.config = config
        self.aucroc = AUROC(task='multiclass', num_classes=config.experiment_params['classes'], average='macro')
        self.average_precision = AveragePrecision(task='multiclass', num_classes=config.experiment_params['classes'], average=None)
    
    def save_predictions_as_csv(self, cache, trainer, col='', save_string='test'):
        print('saving results')
        logger = trainer.default_logger
        y_hat, y = cache.return_results()
        output_folder = os.path.join(logger.save_dir, logger.name, 'version_'+str(logger.version), 'predictions')
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_file_base = 'epoch='+str(trainer.current_epoch)+'-step='+str(trainer.global_step)+'-'

        # truth_values = pd.DataFrame(y.cpu().numpy(), columns=None)
        if (len(y_hat.shape)>1):
            columns = None
        else:
            columns = [col]

        prediction_values = pd.DataFrame(y_hat.numpy(), columns=columns)
        
        prediction_values.to_csv(output_folder + '/' + output_file_base + '_' + save_string +'_predictions'+str(torch.distributed.get_rank())+'.csv', index=False)
        
        try:
            imgs = trainer.datamodule.dataset.inputs_path
            imgs_values = pd.DataFrame(imgs, columns=None)
            imgs_values.to_csv(output_folder + '/' + output_file_base + '_' + save_string +'_imgs.csv', index=False)
        except:
            pass
    
    def on_train_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        dataset = type(pl_module.trainer.datamodule).__name__
        if (hasattr(pl_module, 'metric_probs_train')):
            for key, value in pl_module.metric_cache_train.items():
                preds, target = value.return_results()
                if (trainer.current_epoch == trainer.max_epochs-1):
                    self.save_predictions_as_csv(value, trainer, col=key, save_string=dataset+'_train_'+key)
                value.reset()


    def on_validation_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        for key, value in pl_module.metric_cache_val.items():
            preds, target = value.return_results()
            if (len(preds.shape) > 1 and self.config.prediction_results_calc_AUC): #passing acc directly
                metric = self.aucroc(preds, target)
                self.log('val_'+key+'_AUC_ROC', metric, prog_bar=False, on_epoch=True, on_step=False, logger=True, sync_dist=True)
                metric = self.average_precision(preds, target)
                for i in range(len(metric)):
                    self.log('val_'+key+'_AUC_PR_'+ str(i), metric[i], prog_bar=False, on_epoch=True, on_step=False, logger=True, sync_dist=True)
            value.reset()

    def on_test_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        dataset = type(pl_module.trainer.datamodule).__name__

        for key, value in pl_module.metric_cache.items():
            preds, target = value.return_results()
            if (len(preds.shape) > 1 and self.config.prediction_results_calc_AUC): #passing acc directly
                metric = self.aucroc(preds, target)
                self.log('test_'+key+'_AUC_ROC', metric, prog_bar=False, on_epoch=True, on_step=False, logger=True, sync_dist=True)
                metric = self.average_precision(preds, target)

                for i in range(len(metric)):
                    self.log('test_'+key+'_AUC_PR_'+ str(i), metric[i], prog_bar=False, on_epoch=True, on_step=False, logger=True, sync_dist=True)
            if (len(preds.shape) > 0): #passing acc directly
                self.save_predictions_as_csv(value, trainer, col=key, save_string=dataset+'_test_'+key)
            value.reset()

# =======================================================================================================================
# https://github.com/Lightning-AI/lightning/issues/4836
class UpdateDataloaderEpoch(Callback):
    def on_train_epoch_start(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        if (trainer.strategy.__class__.__name__ == 'DDPStrategy' or trainer.strategy.__class__.__name__ == 'MyDDPStrategy'):
            for key, dataloader in trainer.train_dataloader.loaders.items():
                if (hasattr(dataloader, 'loader')):
                    if hasattr(dataloader.loader.sampler, 'set_epoch'):
                        dataloader.loader.sampler.set_epoch(trainer.current_epoch)
                else:
                    if hasattr(dataloader.sampler, 'set_epoch'):
                        dataloader.sampler.set_epoch(trainer.current_epoch)
        return super().on_train_epoch_start(trainer, pl_module)

    def on_validation_epoch_start(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        if (trainer.strategy.__class__.__name__ == 'DDPStrategy' or trainer.strategy.__class__.__name__ == 'MyDDPStrategy'):
            for dataloader in trainer.val_dataloaders:
                if hasattr(dataloader.sampler, 'set_epoch'):
                    dataloader.sampler.set_epoch(trainer.current_epoch)
        return super().on_validation_epoch_start(trainer, pl_module)
# =======================================================================================================================
