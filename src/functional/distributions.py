import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist

from pdb import set_trace as pb

class CategoricalLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def init(self, input):
        self.dist = dist.Categorical(logits = input)

    def forward(self, target):
        return self.dist.log_prob(target)

class NormalLoss(nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self, target, input):
        loss = dist.Normal(input, torch.ones_like(input)).log_prob(target)
        loss = torch.sum(loss, dim=tuple(1+i for i in range(len(loss.shape)-1)))
        return loss


class BernoulliLoss(nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self, target, input):
        loss = -F.binary_cross_entropy_with_logits(input, target, reduction='none')
        # loss = dist.continuous_bernoulli.ContinuousBernoulli(logits=input).log_prob(target)
        # dist.Bernoulli(logits=input).log_prob(target.round())
        loss = torch.sum(loss, dim=tuple(1+i for i in range(len(loss.shape)-1)))
        return loss


class LaplaceLoss(nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self, target, x):
        loss = dist.Laplace(x, torch.ones_like(x)).log_prob(target)
        loss = torch.sum(loss, dim=tuple(1+i for i in range(len(loss.shape)-1)))
        return loss


class AbsoluteLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, target, input):
        loss = torch.nn.L1Loss(reduction='none')
        loss = loss(input, target)
        loss = torch.sum(loss, dim=tuple(1+i for i in range(len(loss.shape)-1)))
        return loss

# =======================================================

class NormalSample(nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self, mu, sigma, z_dist, transform=False):
        if transform:
            sigma = sigma.mul(0.5).exp_()
        
        if (z_dist == 'normal'):
            z = dist.Normal(mu, sigma).rsample()
        elif (z_dist == 'none'):
            z = mu

        return z