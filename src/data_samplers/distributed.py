from torch.utils.data.distributed import DistributedSampler
import torch

class MyDistributedSampler(DistributedSampler):
    def __init__(self, dataset, shuffle=True, seed=0, drop_last=False, **kwargs):

        if torch.distributed.is_initialized():
            super(MyDistributedSampler, self).__init__(dataset, num_replicas=None, rank=None, shuffle=shuffle, seed=seed, drop_last=drop_last)
        else:
            super(MyDistributedSampler, self).__init__(dataset, num_replicas=1, rank=0, shuffle=shuffle, seed=seed, drop_last=drop_last)
