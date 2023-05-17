import torch
from pdb import set_trace as pb


def log_sum_exp(x, dim=-1):
    # multiply pixel loss for an item
    const = torch.max(x, dim=dim, keepdim=True) # for numerical stability
    const = const.values
    # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    log_sum = torch.exp(x-const)
    log_sum = torch.log(torch.sum(log_sum, dim=dim)) + const.squeeze()
    return(log_sum)

