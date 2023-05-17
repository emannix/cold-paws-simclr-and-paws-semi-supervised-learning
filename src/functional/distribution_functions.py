import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist

from pdb import set_trace as pb

class LossCategorical(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, target, input):
        return F.softmax(input, dim=1).gather(1, target.view(-1,1)).squeeze() 

class LossLogCategorical(nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self, target, input):
        return -F.cross_entropy(input, target, reduction='none')



# =====================================================================

class LossLogLaplace(nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self, target, input):
        loss= dist.Laplace(input, torch.ones_like(input)).log_prob(target)
        dims = tuple(1+i for i in range(len(loss.shape)-1)) 
        loss = torch.sum(loss, dim=dims)
        return loss


def prob_log_normal(input, mu, sigma):
    return -1/2/sigma.pow(2)*(mu-input).pow(2) + \
                  torch.log(1/(sigma*(2*torch.tensor(np.pi))**0.5))


class LossLogNormal(nn.Module):
    def __init__(self, sigma=1):
        super().__init__() 
        self.sigma = torch.tensor(sigma)
    def forward(self, target, input):
        return -1/2/self.sigma.pow(2)*torch.sum((target-input).pow(2), axis=1) + \
                  torch.log(1/(self.sigma*(2*torch.tensor(np.pi))**0.5))

class LossLogBernoulli(nn.Module):
    def __init__(self, reduction='sum'):
        super().__init__()
        self.reduction = reduction

    def forward(self, target, input):

        loss = -F.binary_cross_entropy(input, target, reduction='none')
        if self.reduction == 'sum': # sum over all axes other than the first
            return torch.sum(loss, dim=tuple(1+i for i in range(len(loss.shape)-1)) )
        elif self.reduction == 'none':
            return loss



class LossLogBernoulliLogit(nn.Module):
    def __init__(self, reduction='sum'):
        super().__init__()
        self.reduction = reduction

    def forward(self, target, input):
        loss = -F.binary_cross_entropy_with_logits(input, target, reduction='none')
        if self.reduction == 'sum': # sum over all axes other than the first
            return torch.sum(loss, dim=tuple(1+i for i in range(len(loss.shape)-1)) )
        elif self.reduction == 'none':
            return loss


class LossLogBernoulliOH(nn.Module):
    def __init__(self, reduction='sum', classes=10):
        super().__init__()
        self.reduction = reduction
        self.classes = classes

    def forward(self, target, input):
        one_hot = torch.nn.functional.one_hot(target, num_classes=self.classes).float()
        loss = -F.binary_cross_entropy(F.sigmoid(input), one_hot, reduction='none')
        if self.reduction == 'sum': # sum over all axes other than the first
            return torch.sum(loss, dim=tuple(1+i for i in range(len(loss.shape)-1)) )
        elif self.reduction == 'none':
            return loss

