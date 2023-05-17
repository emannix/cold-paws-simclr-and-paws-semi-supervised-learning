import torch
from torch import Tensor

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
import math

from torch.utils.data import Sampler
from pdb import set_trace as pb
import numpy as np
import torch.distributed as dist

from .base_sampler import _RandomSampler

# ====================================================================================

class RandomUpSampler(_RandomSampler):

    def __init__(self, data_source: Sized,
                 shuffle = True, shuffle_extra=False,
                 batch_size: Optional[int] = 256, 
                 generator=None, upsample_to=10,
                 seed=10,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None) -> None:
        self.upsample_to = upsample_to
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_extra = shuffle_extra

        self.generator = torch.Generator()
        super(RandomUpSampler, self).__init__(data_source, batch_size, self.generator)
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
        self.outer_epoch = 0
        self.seed = seed
        self.rank = rank

    def sample_indices(self, generator):
        # manual_seed = self.seed*1e6 + self.epoch*1e2 + self.rank
        # generator.manual_seed(math.floor(manual_seed))
        # Generate large list, then subsample it for each gpu
        seed = int(self.seed + self.outer_epoch*100) 
        self.rng = np.random.default_rng(int(seed))
        self.rng_shuffler = np.random.default_rng(int(seed))

        n = len(self.data_source)
        num_expand_x = math.ceil(
            self.upsample_to / n)

        if self.shuffle:
            upscaled_index = np.hstack([self.rng.permutation(n) for _ in range(num_expand_x)])
        else:
            upscaled_index = np.hstack([torch.arange(n).numpy() for _ in range(num_expand_x)])

        gpu_portion = self.upsample_to//self.num_replicas
        upscaled_index = upscaled_index[int(self.rank*gpu_portion):int((self.rank+1)*gpu_portion)]
        # print(upscaled_index)
        if self.shuffle_extra:
            self.rng_shuffler.shuffle(upscaled_index)
        # print(self.data_source.indices[upscaled_index])
        return upscaled_index.tolist()

    def __len__(self) -> int:
        n = len(self.data_source)
        num_expand_x = math.ceil(self.upsample_to / n / self.num_replicas)#*self.num_replicas
        return n*num_expand_x

    def set_epoch(self, epoch):
        self.outer_epoch = epoch
