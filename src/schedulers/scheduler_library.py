
import torch
from .linear_learning_rate import LinearLR
from .cosine_annealing import get_cosine_schedule_with_warmup, get_cosine_schedule_with_warmup_group2
from .cosine_annealing_with_warmup import LinearWarmupAndCosineAnneal
from torch.optim import lr_scheduler
from . import paws_warmup_cosine_annealing
from pdb import set_trace as pb


def scheduler_library(self, optimizer):
    if (self.schedule == 'CosineAnnealingLR'):
        scheduler = lr_scheduler.CosineAnnealingLR(
          optimizer = optimizer,
          T_max= self.config.max_epochs,
          eta_min = self.min_learning_rate
        )
        scheduler = {"scheduler": scheduler, "interval" : "epoch" }
    elif (self.schedule == 'get_cosine_schedule_with_warmup'):
        self.trainer.reset_train_dataloader()
        scheduler = get_cosine_schedule_with_warmup(
          optimizer = optimizer,
          num_warmup_steps=0,
          num_training_steps = self.config.max_epochs*self.trainer.num_training_batches
        )
        scheduler = {"scheduler": scheduler, "interval" : "step" }
    elif (self.schedule == 'get_cosine_schedule_with_warmup_group2'):
        self.trainer.reset_train_dataloader()
        scheduler = get_cosine_schedule_with_warmup_group2(
          optimizer = optimizer,
          num_warmup_steps=0,
          num_training_steps = self.config.max_epochs*self.trainer.num_training_batches
        )
        scheduler = {"scheduler": scheduler, "interval" : "step" }
    # ======================================================================
    elif (self.schedule == 'LinearWarmupCosineAnnealingLR'):
        self.trainer.reset_train_dataloader()
        scheduler = LinearWarmupAndCosineAnneal(
          optimizer = optimizer,
          warm_up=self.config.warm_up, last_epoch=-1,
          T_max = self.config.max_epochs*self.trainer.num_training_batches
        )
        scheduler = {"scheduler": scheduler, "interval" : "step" }
    elif (self.schedule == 'LinearLR'):
        self.trainer.reset_train_dataloader()
        scheduler = LinearLR(
          optimizer = optimizer,
          num_epochs=self.config.max_epochs*self.trainer.num_training_batches, last_epoch=-1
        )
        scheduler = {"scheduler": scheduler, "interval" : "step" }
    elif (self.schedule == 'Constant'):
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda x: 1 )
        scheduler = {"scheduler": scheduler, "interval" : "step" }
    elif (self.schedule == 'paws_WarmupCosineSchedule'):
        self.trainer.reset_train_dataloader()
        scheduler = paws_warmup_cosine_annealing.WarmupCosineSchedule(
            optimizer,
            warmup_steps=self.config.warm_up*self.trainer.num_training_batches,
            start_lr=self.start_learning_rate,
            ref_lr=self.learning_rate,
            final_lr=self.final_learning_rate,
            T_max=self.config.max_epochs*self.trainer.num_training_batches)
        scheduler = {"scheduler": scheduler, "interval" : "step" }
        
    return scheduler


