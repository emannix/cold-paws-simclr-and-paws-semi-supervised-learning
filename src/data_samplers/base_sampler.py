import torch
from torch import Tensor

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

from torch.utils.data import Sampler
from pdb import set_trace as pb
import numpy as np

# https://github.com/pytorch/pytorch/blob/master/torch/utils/data/sampler.py
class _RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized

    def __init__(self, data_source: Sized,
                 batch_size: Optional[int] = 256, 
                 generator=None, **kwargs) -> None:
        self.data_source = data_source
        self.generator = generator
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        indices = self.sample_indices(generator)
        for _ in range(self.__len__()):
            yield from indices

