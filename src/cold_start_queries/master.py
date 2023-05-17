import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from copy import deepcopy

from pdb import set_trace as pb

from pathlib import Path
import os
import pandas as pd
import glob
import yaml


def create_query(trainer, model, datamodule, select_indices_query, number_of_samples): # self.active_budget
        trainer.predict_loop.epoch_loop.return_predictions = True
        trainer.predict_loop._return_predictions: bool = True

        dataset_labelled = datamodule.dataset_labelled
        dataset_unlabelled = datamodule.dataset_unlabelled

        if select_indices_query in ['load_indices_vector']: # queries saved as index files
            picked_indices_csv = pd.read_csv(model.init_data_params.load_indices_csv)
            picked_indices = picked_indices_csv['indices'].values
            # ==============================================================================================================
        elif select_indices_query in ['evaluate_representations']: # compute and save representations
            datamodule.predict_dataset = 'unlabelled'
            trainer.reset_predict_dataloader(model)
            predictions = trainer.predict_loop.run()

            picked_indices = np.arange(20)
            # ==============================================================================================================

        else:
            raise NameError('Please select a valid query')

        label_values = pd.DataFrame(dataset_unlabelled.dataset.targets[picked_indices].cpu().numpy(), columns=['label'])
        datamodule.move_unlabelled_to_labelled(picked_indices, get_index=False)
        # ================================================
        index_values = pd.DataFrame(picked_indices, columns=['index'])
        df = pd.concat([index_values.reset_index(drop=True),label_values.reset_index(drop=True)], axis=1)

        if torch.distributed.is_initialized():
            rank = str(torch.distributed.get_rank())
        else:
            rank = '0'

        logger = trainer.default_logger
        output_folder = os.path.join(logger.save_dir, logger.name, 'version_'+str(logger.version), 'predictions')
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_file_base = 'epoch='+str(trainer.current_epoch)+'-step='+str(trainer.global_step)+'-'
        df.to_csv(output_folder + '/' + output_file_base + '_active_learning'+rank+'.csv', index=False)
