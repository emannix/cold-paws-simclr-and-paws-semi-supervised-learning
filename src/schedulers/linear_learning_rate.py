import numpy as np
import torch
import warnings
import time
import torch.distributed as dist

from pdb import set_trace as pb
# https://github.com/AndrewAtanov/simclr-pytorch/blob/master/utils/utils.py


class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, last_epoch=-1):
        self.num_epochs = max(num_epochs, 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        res = []
        for lr in self.base_lrs:
            res.append(np.maximum(lr * np.minimum(-self.last_epoch * 1. / self.num_epochs + 1., 1.), 0.))
        return res
        