
import torch
from .lars_optimizer import LARS
from .SGD import SGD
from . import paws_lars
from . import paws_SGD
from pdb import set_trace as pb


def optimizer_library(self, grouped_parameters):
    if (self.optimizer == 'SGD'):
        optimizer = torch.optim.SGD(grouped_parameters, 
            lr=self.learning_rate, weight_decay=self.weight_decay,
            momentum = self.momentum, nesterov=True)
    elif (self.optimizer == 'SGD-nonesterov'):
        optimizer = torch.optim.SGD(grouped_parameters, 
            lr=self.learning_rate, weight_decay=self.weight_decay,
            momentum = self.momentum, nesterov=False)
    elif (self.optimizer == 'Adam'):
        optimizer = torch.optim.Adam(grouped_parameters, 
                lr=self.learning_rate, weight_decay=self.weight_decay
                )
    elif (self.optimizer == 'SGD-fixmatch'):
        optimizer = torch.optim.SGD(grouped_parameters, 
            lr=self.learning_rate,
            momentum = self.momentum, nesterov=True)
    elif (self.optimizer == 'SGD-simclr'):
        optimizer = torch.optim.SGD(grouped_parameters, 
            lr=self.learning_rate,
            momentum = self.momentum)
    elif (self.optimizer == 'LARS'):
        sgd_optimizer = torch.optim.SGD(grouped_parameters, 
            lr=self.learning_rate,
            momentum = self.momentum)
        optimizer = LARS(sgd_optimizer)
# ================================================================================
    elif (self.optimizer == 'mySGD'):
    	optimizer = SGD(grouped_parameters, 
            lr=self.learning_rate, nesterov=self.momentum_nesterov,
            momentum_version=self.momentum_version, weight_decay=self.weight_decay,
            momentum = self.momentum)
    elif (self.optimizer == 'mySGDLARS'):
        sgd_optimizer = SGD(grouped_parameters, 
            lr=self.learning_rate, nesterov=self.momentum_nesterov,
            momentum_version=self.momentum_version, weight_decay=self.weight_decay,
            momentum = self.momentum)
        optimizer = LARS(sgd_optimizer)
    elif (self.optimizer == 'pawsSGDLARS'):
        optimizer = paws_SGD.SGD(
            grouped_parameters,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=self.momentum_nesterov,
            lr=self.learning_rate)
        optimizer = paws_lars.LARS(optimizer, trust_coefficient=0.001)
    return optimizer


