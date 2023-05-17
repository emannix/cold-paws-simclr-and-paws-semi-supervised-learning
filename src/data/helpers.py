import torch
from sklearn.model_selection import StratifiedKFold
from pdb import set_trace as pb

from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import copy

def sample_stratified(dataset = torch.utils.data.dataset, n_splits = 5):

    x = list(range(0, len(dataset)))
    y = dataset.targets

    skf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=True)

    train_index, test_index = next(skf.split(x, y))
    return train_index, test_index

def sample_balanced_by_class(dataset = torch.utils.data.dataset, per_class = 100):
    x = torch.tensor(list(range(0, len(dataset))))
    if isinstance(dataset, (torch.utils.data.dataset.Subset)):
        if not torch.is_tensor(dataset.dataset.targets):
            dataset.dataset.targets = torch.tensor(dataset.dataset.targets)
        y = dataset.dataset.targets[dataset.indices]
    else:
        y = dataset.targets

    if not torch.is_tensor(y):
        y = torch.tensor(y)
    classes = torch.unique(y)

    balanced_index = []
    for i in range(len(classes)):
        y_p = y == classes[i]
        y_i = np.random.choice(x[y_p], size=per_class, replace=False)
        balanced_index.append(y_i)

    balanced_index = np.concatenate(balanced_index)
    return balanced_index

def sample_unbalanced(dataset = torch.utils.data.dataset, 
    labelled_items = 100, total_items = 1000,
    inputs='inputs', targets='targets'):
    x = torch.tensor(list(range(0, len(dataset))))

    y_total = np.random.choice(x, size=total_items, replace=False)
    y_labelled = np.random.choice(y_total, size=labelled_items, replace=False)
    y_unlabelled = set(y_total) - set(y_labelled)
    y_unlabelled = np.array(list(y_unlabelled)).astype(int)

    train_labelled = Subset(copy.deepcopy(dataset), y_labelled)
    train_labelled = extract_subset_dataset(train_labelled, inputs, targets)

    train_unlabelled = Subset(copy.deepcopy(dataset), y_unlabelled)
    train_unlabelled = extract_subset_dataset(train_unlabelled, inputs, targets)
    
    return train_labelled, train_unlabelled


def extract_subset_dataset(subset_dataset, obj_data = 'train_labels', obj_target = 'targets'):
    # loader = DataLoader(subset_dataset, batch_size=len(subset_dataset.indices),
    #            shuffle=False, num_workers=0, drop_last=False)
    # res = next(iter(loader))

    output = subset_dataset
    if isinstance(output, (torch.utils.data.dataset.ConcatDataset)):
        output_init = copy.deepcopy(output.datasets[0])
        for i in range(1,len(output.datasets)):
            try:
                setattr(output_init, obj_data, 
                        torch.cat(( getattr(output_init, obj_data), 
                            getattr(output.datasets[i], obj_data) ))
                        )
            except:
                setattr(output_init, obj_data, 
                        np.concatenate(( getattr(output_init, obj_data), 
                            getattr(output.datasets[i], obj_data) ))
                        )
            setattr(output_init, obj_target, 
                    torch.cat(( getattr(output_init, obj_target), 
                        getattr(output.datasets[i], obj_target) ))
                    )
        output = output_init
    else:
        output = subset_dataset.dataset
        data_object = getattr(output, obj_data)
        if isinstance(data_object, list):
            setattr(output, obj_data, [data_object[i] for i in subset_dataset.indices])
        else:
            setattr(output, obj_data, data_object[subset_dataset.indices])

        target_object = getattr(output, obj_target)
        if not torch.is_tensor(target_object):
            target_object = torch.tensor(target_object)
        setattr(output, obj_target, target_object[subset_dataset.indices])

    setattr(output, obj_target, getattr(output, obj_target).long())
    return output

# https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
def calculate_class_weights(subset_dataset, classes):
        targets = subset_dataset.dataset.targets[subset_dataset.indices]

        class_counts = []
        for t in range(classes):
            class_counts.append(torch.sum(targets == t).item())

        class_counts = np.array(class_counts)
        class_counts[class_counts == 0] = 1 # if zero examples, set to 1

        # ==========================================
        beta = 0.9999

        class_weights = (1-beta)/(1-beta**class_counts)
        class_weights = class_weights/np.sum(class_weights)*classes
        class_weights = torch.tensor(class_weights)

        return class_weights