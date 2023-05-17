
import torch
import numpy as np
from pdb import set_trace as pb
from torch.optim.lr_scheduler import LambdaLR
import math

class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, lr_max, lr_min, warmup_epochs, max_epochs):
        self.lr_max, self.lr_min = lr_max, lr_min
        self.warmup_epochs, self.max_epochs = warmup_epochs, max_epochs
        self.current_epoch = 0
        super().__init__(optimizer)

    def get_lr(self):
        epoch = self.current_epoch
        if (epoch < self.warmup_epochs):
            lr =  self.lr_min # + (self.lr_max - self.lr_min) * epoch/self.warmup_epochs
        else:
            epoch = epoch - self.warmup_epochs
            max_epochs = self.max_epochs - self.warmup_epochs
            lr =  self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (1 + np.cos(epoch / self.max_epochs * np.pi))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.lr = lr
        self.current_epoch += 1
        return [self.lr]

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup_group2(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    def _lr_lambda2(current_step):
        return 1.0

    return LambdaLR(optimizer, [_lr_lambda2, _lr_lambda], last_epoch)