import torch
from torch import Tensor

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

from torch.utils.data import Sampler
from pdb import set_trace as pb
import numpy as np
import math

from .base_sampler import _RandomSampler

# ====================================================================================

class RandomSamplerPairProb(_RandomSampler):

    def sample_indices(self, generator):

        n = len(self.data_source)
        half_batch = self.batch_size//2
        batches = n//half_batch

        permuted_index = torch.randperm(n, generator=generator).tolist()

        indices = [torch.arange(i*half_batch, (i+1)*half_batch, dtype=torch.int64).tolist() + \
                   torch.randint(low=i*half_batch,high=(i+1)*half_batch, size=(half_batch,), dtype=torch.int64, generator=generator).tolist() \
                 for i in range(batches)]
        indices = np.concatenate(indices)

        permuted_index = np.array(permuted_index)
        
        return permuted_index[indices].tolist()

    def __len__(self) -> int:
        n = len(self.data_source)
        half_batch = self.batch_size//2
        return (n//half_batch)*half_batch*2

# ====================================================================================
# ====================================================================================

class RandomSamplerZPoints(_RandomSampler):

    def __init__(self, data_source: Sized,
                 batch_size: Optional[int] = 256, 
                 generator=None, z_points=10, upsample_to=1000) -> None:
        self.z_points = z_points
        self.upsample_to = upsample_to//self.z_points
        super(RandomSamplerZPoints, self).__init__(data_source, batch_size, generator)
        self.indices = None


    def sample_indices(self, generator):

        n = len(self.data_source)

        # permuted_index = torch.randperm(n, generator=generator).tolist()
        index = np.arange(n)

        num_expand_x = math.ceil(
            self.upsample_to / n)
        upscaled_index = np.hstack([np.random.permutation(index) for _ in range(num_expand_x)])
        upscaled_index = upscaled_index.tolist()

        indices = [ [i]*self.z_points \
                 for i in upscaled_index]
        indices = np.concatenate(indices) 
        self.indices = indices

        return indices.tolist()

    def __len__(self) -> int:
        n = len(self.data_source)
        num_expand_x = math.ceil(self.upsample_to / n)
        return n*num_expand_x*self.z_points
        # return self.upsample_to*self.z_points