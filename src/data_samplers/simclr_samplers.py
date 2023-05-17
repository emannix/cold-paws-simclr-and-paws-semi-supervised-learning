import torch
from torch import Tensor

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

from torch.utils.data import Sampler
from pdb import set_trace as pb
import numpy as np
import math

import torch.distributed as dist

from .base_sampler import _RandomSampler

# ====================================================================================
# https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler

class RandomSamplerPair(_RandomSampler):
    def __init__(self, data_source: Sized,
                 shuffle = True,
                 batch_size: Optional[int] = 256, 
                 seed: Optional[int] = 0, 
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.num_replicas = num_replicas
        self.rank = rank
        self.outer_epoch = 0
        self.seed = seed
        self.drop_last = True
        self.shuffle = shuffle

        self.generator = torch.Generator()
        super(RandomSamplerPair, self).__init__(data_source, batch_size, self.generator)

        if len(self.data_source) % self.num_replicas != 0:  # type: ignore[arg-type]
            self.num_samples = math.ceil(
                (len(self.data_source) - self.num_replicas) / self.num_replicas  
            )
        else:
            self.num_samples = math.ceil(len(self.data_source) / self.num_replicas)  
        self.total_size = self.num_samples * self.num_replicas

        self.rng = np.random.default_rng(self.seed)

    def sample_indices(self, generator):
        # rng = np.random.default_rng(self.seed + self.epoch)
        # generator.manual_seed(self.seed + self.epoch+2)
        # each sampler generates the same permutation, per epoch, and this is downsampled for each gpu
        seed = int(self.seed + self.outer_epoch*100) 
        self.rng = np.random.default_rng(seed)

        if self.shuffle:
            permuted_index = self.rng.permutation(len(self.data_source))
            # permuted_index = torch.randperm(len(self.data_source), generator=None).tolist() # generator
        else:
            permuted_index = torch.arange(len(self.data_source)).tolist()

        permuted_index = permuted_index[self.rank:self.total_size:self.num_replicas]

        n = len(permuted_index)

        half_batch = self.batch_size//2

        batches = n//half_batch
        indices = [torch.arange(i*half_batch, (i+1)*half_batch, dtype=torch.int64).tolist() + \
                   torch.arange(i*half_batch, (i+1)*half_batch, dtype=torch.int64).tolist() \
                 for i in range(batches)]
        indices = np.concatenate(indices)

        permuted_index = np.array(permuted_index)
        
        return permuted_index[indices].tolist()

    def __len__(self) -> int:
        return self.num_samples*2

    def set_epoch(self, epoch):
        self.outer_epoch = epoch