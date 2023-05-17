import torch
from torch import Tensor

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
import math

from torch.utils.data import Sampler
from pdb import set_trace as pb
import numpy as np

from .base_sampler import _RandomSampler

# ====================================================================================

class ImbalanceUpSampler(_RandomSampler):

    def __init__(self, data_source: Sized,
                 batch_size: Optional[int] = 256, 
                 generator=None, upsample_to=10) -> None:
        self.upsample_to = upsample_to
        self.batch_size = batch_size
        super(ImbalanceUpSampler, self).__init__(data_source, batch_size, generator)


    def sample_indices(self, generator):
        n = len(self.data_source)
        index = np.arange(n)
        targets = self.data_source.dataset.targets[self.data_source.indices]

        num_expand_x = math.ceil(
            self.upsample_to / n)
        # upscaled_index = np.hstack([index for _ in range(num_expand_x)])
        # np.random.shuffle(upscaled_index)

        class_counts = []
        for t in np.unique(targets):
            class_counts.append(torch.sum(targets == t).item())
        class_counts = np.array(class_counts)
        total_classes = np.sum(class_counts)

        class_weights = total_classes/class_counts
        item_weights = class_weights[targets]
        item_weights = item_weights/np.sum(item_weights)

        upscaled_index = np.random.choice(index, size=self.upsample_to, replace=True, p=item_weights)
        return upscaled_index.tolist()

    def __len__(self) -> int:
        return self.upsample_to

