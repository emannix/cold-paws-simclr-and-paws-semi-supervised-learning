import torch
import numpy as np
from .helpers import extract_subset_dataset
from pdb import set_trace as pb
import copy
from torch.utils.data import DataLoader, random_split, Subset

def parse_data_helper(train, test, exclude_labelled, 
                unlabelled_samples, labelled_samples, validation_samples,
                data_str='inputs', label_str = 'targets'):
    if exclude_labelled:
        total_training = unlabelled_samples+labelled_samples
    else:
        total_training = max(unlabelled_samples, labelled_samples)

    if total_training < len(train) and unlabelled_samples > 0:
        indices = np.random.choice(np.arange(len(train)), size=total_training, replace=False)

        training = Subset(train, indices)
        training = copy.deepcopy(training)
        training = extract_subset_dataset(training, data_str, label_str)
    else:
        training = train
        if not torch.is_tensor(training.targets):
            training.targets = torch.tensor(training.targets)

    if validation_samples == 0:
        validation = test
        validation_missing_size = None
    else:
        validation = None
        validation_missing_size = validation_samples
    return training, validation, validation_missing_size