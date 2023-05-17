import argparse
import os
import glob
import torch
import shutil
import sys

import torch
import random
import numpy as np
import pickle
import copy

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from pycallbacks import SaveResults, UpdateDataloaderEpoch
from src.callbacks import EMA, SaveRepresentation, LoadInitData

from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.profiler import AdvancedProfiler

import uuid
from pytorch_lightning.loops import FitLoop, TrainingEpochLoop, TrainingBatchLoop, OptimizerLoop, EvaluationLoop, PredictionLoop

from src.utils import MyDDPStrategy
import time
import traceback

import logging

from pdb import set_trace as pb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--test-only", default=False, action='store_true')
    args = parser.parse_args()
    return args

def import_config():
    args = parse_args()
    if (args.config.endswith('.py')):
      sys.path.append(os.path.dirname(args.config))
      config = getattr(__import__(os.path.basename(args.config.replace('.py', '')), fromlist=['Config']), 'Config')
    elif (args.config.endswith('.pkl')):
      config = pickle.load( open( args.config, "rb" ) )
    return(config, args.test_only)

# =========================================================================

def build_trainer(config, checkpoint = True, fitloop=True):

  if not os.path.exists(config.output_dir): os.makedirs(config.output_dir)

  default_logger = TensorBoardLogger(config.output_dir, name=config.run_name, version=config.version)
  if hasattr(config, 'logger') and config.logger is False:
    config.logger = False
    default_logger = False
  else:
    config.logger = default_logger

  config.reload_dataloaders_every_n_epochs = int(0)
  config.enable_checkpointing = False
  config.log_every_n_steps=20

  # ========================================================
  config_use = copy.deepcopy(config)
  if (hasattr(config, 'strategy')):
    if config.strategy == 'myddp':
      config_use.strategy = MyDDPStrategy()
  # ========================================================
  config_use.enable_progress_bar = False
  # ========================================================
  config_use.devices = config_use.gpus
  config_use.accelerator = 'gpu'
  config_use.gpus = None

  trainer = Trainer.from_argparse_args(config_use)

  # ========================================================
  if (hasattr(config, 'ema_decay')):
    trainer.callbacks.append(EMA(decay=config.ema_decay))
  if "save_representation" in config.experiment_params:
    trainer.callbacks.append(SaveRepresentation(config))
  # ========================================================
  if (hasattr(config, 'early_stopping_patience')):
    if config.early_stopping_patience != -1:
      trainer.callbacks.append(EarlyStopping(
          monitor=config.metric_monitor, mode=config.metric_monitor_mode,
          patience=config.early_stopping_patience
        ))
  # ========================================================

  trainer.default_logger = default_logger

  if (hasattr(config, 'save_top_k')):
    save_top_k = config.save_top_k
  else:
    save_top_k = 1
  
  if checkpoint:
    checkpoint_callback = ModelCheckpoint(monitor=config.metric_monitor, every_n_epochs=1,
                        save_top_k=save_top_k, save_last=True, mode=config.metric_monitor_mode,
                        save_weights_only = False)
    trainer.callbacks.append(checkpoint_callback)
    
  progress_callback = TQDMProgressBar(refresh_rate=20)
  lr_log_callback = LearningRateMonitor(logging_interval='step', log_momentum=True)
  trainer.callbacks.append(lr_log_callback)

  if hasattr(config, 'save_prediction_results'):
    if (config.save_prediction_results):
      trainer.callbacks.append(SaveResults(config))
  else:
      config.prediction_results_calc_AUC = True
      trainer.callbacks.append(SaveResults(config))

  trainer.callbacks.append(LoadInitData())
  trainer.callbacks.append(UpdateDataloaderEpoch())
  trainer.callbacks.append(progress_callback)
  return(trainer)

def build_data(config):
  config.data_params['workers'] = config.num_workers
  dat = config.data_class(**config.data_params)
  return dat

def build_model(config):
  
  if hasattr(config, 'run_hash'):
    config.run_hash = uuid.uuid4().hex

  config.data_name = config.data_class.__name__
  config.experiment_name = config.experiment_class.__name__
  
  model = config.experiment_class(config)
  if hasattr(config, 'deterministic_params'):
    model.reset_parameters(config.deterministic_params)


  return model

def train_with_config(config):
  shutil.rmtree( os.path.join(config.output_dir, config.run_name, 'version_'+str(config.version)), 
                    ignore_errors=True )

  dat = build_data(config)
  trainer = build_trainer(config)
  config.test = False
  model = build_model(config)

  trainer.fit(model, dat)
  # make sure the last checkpoint is always saved, even when validation and training do not run

  check_callbacks = [isinstance(obj, ModelCheckpoint) for obj in trainer.callbacks]
  checkpoint_ind = np.where(check_callbacks)[0][0]

  if trainer.callbacks[checkpoint_ind].last_model_path == '':
    monitor_candidates = trainer.callbacks[checkpoint_ind]._monitor_candidates(trainer)
    trainer.callbacks[checkpoint_ind]._save_last_checkpoint(trainer, monitor_candidates)
  # =========================================

  model_checkpoints = [*trainer.callbacks[checkpoint_ind].best_k_models] + [trainer.callbacks[checkpoint_ind].last_model_path]

  return dat, model_checkpoints
  
def test_with_config(config, dat, model_checkpoints = None):
  print(os.path.dirname(os.path.realpath(__file__)))
  if model_checkpoints is None:
    checkpoint_dir = os.path.join(config.output_dir, config.run_name, 'version_'+str(config.version), 'checkpoints')
    model_checkpoints = os.listdir(checkpoint_dir)
    model_checkpoints = [os.path.join(checkpoint_dir,x) for x in model_checkpoints]

  for i in range(len(model_checkpoints)):
    checkpoint = model_checkpoints[i]
    print(checkpoint)

    config.test = True
    # =====================================
    counter = 0
    while True:
      try:
        trainer = Trainer(max_epochs=0, accelerator='cpu')
        _loaded_checkpoint = trainer._checkpoint_connector._load_and_validate_checkpoint(checkpoint)
        config.max_epochs = _loaded_checkpoint['epoch']
        break
      except Exception:
        time.sleep(1)
        counter += 1
      if counter > 10:
        print("Taking too long to load") 
        sys.exit(1)
    # =====================================

    model_best = build_model(config)
    trainer = build_trainer(config, checkpoint=False, fitloop=False)
    trainer.fit(model_best, dat, ckpt_path=checkpoint) # automatically restores model, epoch, step, LR schedulers, apex, etc...
    trainer.test(model_best, datamodule=dat, ckpt_path=checkpoint)

if __name__ == "__main__":
    # ===========================================
    # ===========================================

    config, test_only = import_config()
    print(config)

    # ==========================================================
    # Job array management
    base_seed = config.seed
    if os.environ.get('PBS_ARRAY_INDEX') is not None:
      array_idx = int(os.environ.get('PBS_ARRAY_INDEX')) - base_seed
    elif os.environ.get('SLURM_ARRAY_TASK_ID') is not None:
      array_idx = int(os.environ.get('SLURM_ARRAY_TASK_ID')) - base_seed
    else:
      array_idx = 0

    config.seed = base_seed + array_idx
    config.run_name = config.run_name + '_seed' + str(config.seed)
    if not hasattr(config, 'test_function'):
      if 'sampler_conf' in config.data_params:
        if 'seed' in config.data_params['sampler_conf']:
          config.data_params['sampler_conf']['seed'] = config.seed
      if 'sampler_val_conf' in config.data_params:
        if 'seed' in config.data_params['sampler_val_conf']:
          config.data_params['sampler_val_conf']['seed'] = config.seed
      if 'sampler_unlabelled_conf' in config.data_params:
        if 'seed' in config.data_params['sampler_unlabelled_conf']:
          config.data_params['sampler_unlabelled_conf']['seed'] = config.seed

    if hasattr(config, 'initial_selection_load_indices_path'):
      folder_search = config.initial_selection_load_indices_path + config.initial_selection_load_indices_folder
      files_search = glob.glob(folder_search+"/X*")
      files_search.sort()
      config.data_params['selected_labels']['load_indices_csv'] = \
          os.path.relpath(files_search[config.initial_selection_start_ind + array_idx])

    # ==========================================================

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    seed_everything(config.seed)
    if config.precision == 64:
      torch.set_default_dtype(torch.float64)

    if not test_only:
      dat, model_checkpoints = train_with_config(config)
    else:
      dat = build_data(config)
      model_checkpoints = None
    test_with_config(config, dat, model_checkpoints)
