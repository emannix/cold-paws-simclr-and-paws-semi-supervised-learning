
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from collections import OrderedDict


class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=128, init_seed=-1):
        super(ProjectionHead, self).__init__()

        if init_seed != -1:
            torch.manual_seed(init_seed)
            np.random.seed(init_seed)
            random.seed(init_seed)
            torch.cuda.manual_seed_all(init_seed)

        self.fc = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(hidden_dim, hidden_dim)),
            ('bn1', torch.nn.BatchNorm1d(hidden_dim)),
            ('relu1', torch.nn.ReLU(inplace=True)),
            ('fc2', torch.nn.Linear(hidden_dim, hidden_dim)),
            ('bn2', torch.nn.BatchNorm1d(hidden_dim)),
            ('relu2', torch.nn.ReLU(inplace=True)),
            ('fc3', torch.nn.Linear(hidden_dim, output_dim))
        ]))

    def forward(self, x, **kwargs):
        x = self.fc(x)
        return x

class PredictionHead(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=128):
        super(PredictionHead, self).__init__()
        pred_head = OrderedDict([])
        pred_head['bn1'] = torch.nn.BatchNorm1d(output_dim)
        pred_head['fc1'] = torch.nn.Linear(output_dim, output_dim//mx)
        pred_head['bn2'] = torch.nn.BatchNorm1d(output_dim//mx)
        pred_head['relu'] = torch.nn.ReLU(inplace=True)
        pred_head['fc2'] = torch.nn.Linear(output_dim//mx, output_dim)
        self.pred = torch.nn.Sequential(pred_head)

    def forward(self, x, **kwargs):
        x = self.pred(x)
        return x