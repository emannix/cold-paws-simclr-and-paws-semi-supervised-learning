# https://github.com/AndrewAtanov/simclr-pytorch/blob/master/models/losses.py


import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

from ..utils import AllGather

from sklearn.manifold._t_sne import _joint_probabilities, _joint_probabilities_nn
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix

from pdb import set_trace as pb


def accuracy(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    acc = (topk == labels).all(1).float()
    return acc

class NTXent(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUM = 1e4

    def __init__(self, temperature=1.0, multiplier=2, distributed=False, double_precision=False):
        super().__init__()
        self.tau = temperature
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.
        self.double_precision = double_precision

        # =============================================
        # https://github.com/google-research/simclr/blob/master/tf2/objective.py
    def forward(self, z, y):
        n = z.shape[0]
        assert n % self.multiplier == 0

        # np.random.seed(0)
        # imarray = np.random.rand(512,64).astype(dtype="float32")
        # z = torch.tensor(imarray)
        if self.double_precision:
            z = z.double()

        z = F.normalize(z, p=2, dim=1)
        batch_size = z.shape[0]//2
        z1, z2 = torch.split(z, batch_size)

        if self.distributed: # self.distributed
            z1_large = AllGather.apply(z1)
            z2_large = AllGather.apply(z2)

            enlarged_batch_size = z1_large.shape[0]
            labels_oh = torch.arange(batch_size, device=z1.device) + dist.get_rank() * batch_size
            masks = torch.nn.functional.one_hot(labels_oh, enlarged_batch_size).to(z1)
        else:
            z1_large = z1
            z2_large = z2

            # labels = torch.nn.functional.one_hot(torch.arange(batch_size), batch_size * 2).to(z1)
            labels_oh = torch.arange(batch_size, device=z1.device)
            masks = torch.nn.functional.one_hot(torch.arange(batch_size), batch_size).to(z1)
        
        logits_11 = torch.matmul(z1, z1_large.t())/self.tau
        logits_22 = torch.matmul(z2, z2_large.t())/self.tau
        logits_12 = torch.matmul(z1, z2_large.t())/self.tau
        logits_21 = torch.matmul(z2, z1_large.t())/self.tau

        logits_11 = logits_11 - masks * self.LARGE_NUM
        logits_22 = logits_22 - masks * self.LARGE_NUM

        labels = labels_oh
        
        loss_1 = torch.nn.functional.cross_entropy(torch.cat([logits_12, logits_11], 1), labels, reduction='none')
        loss_2 = torch.nn.functional.cross_entropy(torch.cat([logits_21, logits_22], 1), labels, reduction='none')
        loss = -(loss_1 + loss_2)

        # print('====================================')
        # print(-loss.mean().detach().cpu().numpy())
        # print(-loss.max().detach().cpu().numpy())
        # print(-loss.min().detach().cpu().numpy())
        # pb()
        if self.double_precision:
            loss = loss.float()

        acc = torch.Tensor.float(labels_oh == torch.argmax(logits_12, axis=1)) # calculating accuracy
        # print(acc.mean())
        # pb()
        return loss, acc