import torchmetrics
import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from pytorch_lightning.callbacks import Callback

from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
import os
import pandas as pd
import glob

from torchmetrics import AUROC, AveragePrecision
import torch
import numpy as np

from pdb import set_trace as pb

class RepresentationCache(Metric):
    full_state_update=True
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.record = False
        self.representation = []

    def hook(self, model, input, output):
        self.update(output)

    def update(self, representation: torch.Tensor):
        if self.record:
            self.representation.append(representation.detach().cpu())

    def reset(self):
        self.representation = []
        # self.preds_raw = []
        
    def return_results(self):
        # preds_raw = dim_zero_cat(self.preds_raw).squeeze()

        if (len(self.representation) > 0):
            representation = dim_zero_cat(self.representation).squeeze()
        else:
            representation = torch.tensor((0.0, 0.0))

        return(representation)
    
    def compute(self):
        pass

# ==================================================================
# ==================================================================

class SaveRepresentation(Callback):

    def __init__(self, config):
        super().__init__()

    def save_predictions_as_csv(self, cache, trainer, key, save_string):
        print('saving results')
        logger = trainer.default_logger
        representation = cache.return_results()
        output_folder = os.path.join(logger.save_dir, logger.name, 'version_'+str(logger.version), 'predictions')
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_file_base = 'epoch='+str(trainer.current_epoch)+'-step='+str(trainer.global_step)+'-'

        output_file_final = output_folder + '/' + output_file_base + save_string +key+'_' +str(torch.distributed.get_rank())

        if len(representation.shape) > 2:
            representation = representation.cpu().numpy()
            representation.dump(output_file_final + '.npy')
        else:
            if len(representation.shape) == 1:
                columns = [key]
            elif len(representation.shape) == 2:
                columns = None
            representation_values = pd.DataFrame(representation.cpu().numpy(), columns=columns)

            representation_values.to_csv(output_file_final + '.csv', index=False)

    # ==========================================================
    def _epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        if (hasattr(pl_module, 'representation_cache')):
            for key, value in pl_module.representation_cache.items():
                value.record = True

    def on_test_epoch_start(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        self._epoch_start(trainer, pl_module)

    def on_predict_epoch_start(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        self._epoch_start(trainer, pl_module)

    # ==========================================================
    def _epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', save_string='_test_'):
        if (hasattr(pl_module, 'representation_cache')):
            for key, value in pl_module.representation_cache.items():
                self.save_predictions_as_csv(value, trainer, key, save_string)
                value.reset()
                value.record = False

    def on_test_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        self._epoch_end(trainer, pl_module, '_test_')

    def on_predict_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', results
    ):
        self._epoch_end(trainer, pl_module, '_cold_start_')
