import torch
import numpy as np
from pdb import set_trace as pb

import pandas as pd
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os.path

from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union

class ImageFolderCSVDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, image_folder='test/', transform=None,
        metadata_csv_file='test.csv', metadata_csv_image = 'image_id', 
        metadata_csv_label = 'class_id', image_ext = '.png',
        return_index = False):
        'Initialization'
        
        if metadata_csv_file is not None:
            data_sheet = pd.read_csv(metadata_csv_file)
            if isinstance(metadata_csv_image, list):
              cols = data_sheet[metadata_csv_image].agg(''.join, axis=1)+image_ext
            else:
              cols = data_sheet[metadata_csv_image] +image_ext
            imgs = [os.path.join(image_folder, x) for x in cols]
        else:
            imgs = os.listdir(image_folder)
            imgs = [x for x in imgs if image_ext in x]

        if metadata_csv_label is None:
            labels = np.zeros(len(imgs))
        else:
            labels = data_sheet[metadata_csv_label].values

        if labels.dtype.char == 'O':
          self.target_labels = np.unique(labels) # return_counts=True
          labels = np.argwhere(labels[:, None] == self.target_labels[None, :])[:,1]

        self.transform = transform
        self.inputs = np.array(imgs)
        self.targets = torch.tensor(labels)
        self.return_index = return_index
 
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)

  def __getitem__(self, index):
        'Generates one sample of data'
        target = self.targets[index]
        img_path = self.inputs[index]
        # image = read_image(img_path)
        image = pil_loader(img_path)
        if self.transform is not None:
            image = self.transform(image)
        if (self.return_index):
          return image, target, index
        else:
          return image, target # just grab the image, don't get anything else

# ========================================================

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageFolder(DatasetFolder):

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = pil_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        return_index = False
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.inputs = self.samples
        self.targets = torch.tensor(self.targets)
        self.classes = np.array(self.classes)
        self.return_index = return_index

    def __len__(self):
          'Denotes the total number of samples'
          return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # print(sample.shape)
        # if sample.shape[0] != 3:
        #   pb()

        if (self.return_index):
          return sample, target, index
        else:
          return sample, target # just grab the image, don't get anything else

# ========================================================

class TransImageDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset,
        multicrop_transform=(0, None),
        supervised_views=1,
        target_transform = None,
        return_index = True
    ):
        self.dataset = dataset
        self.targets = self.dataset.targets
        self.classes = np.unique(self.targets[~np.isnan(np.array(self.targets))])
        self.classes = self.classes[self.classes >= 0] # remove negative class indices

        self.return_index=return_index

        self.supervised_views = supervised_views
        self.multicrop_transform = multicrop_transform
        self.target_transform = target_transform

        try:
            self.transform = self.dataset.transform
            self.dataset.transform = None
        except:
            self.transform = self.dataset.dataset.transform
            self.dataset.dataset.transform = None


    def set_supervised(self, labelled_indices):
        mint = None
        self.target_indices = []
        effective_classes = 0
        for t in range(len(self.classes)):
            indices = np.squeeze(np.argwhere(self.targets[labelled_indices] == t)).tolist()
            if not isinstance(indices, list): # this takes care of instances when only one index for a class. It will still fail if there are not single class instances
                indices = [indices]
            # assert len(indices) > 0
            if len(indices) > 0:
                # indices = labelled_indices[indices]
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(mint, len(indices))
                effective_classes += 1
        return effective_classes


    def __getitem__(self, index):
        if self.return_index:
            img, target, idx = self.dataset.__getitem__(index)
        else:
            img, target = self.dataset.__getitem__(index)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:

            if self.supervised:
                return *[self.transform(img) for _ in range(self.supervised_views)], target
            else:
                img_1 = self.transform(img)
                img_2 = self.transform(img)

                multicrop, mc_transform = self.multicrop_transform
                if multicrop > 0 and mc_transform is not None:
                    mc_imgs = [mc_transform(img) for _ in range(int(multicrop))]
                    return img_1, img_2, *mc_imgs, target

                return img_1, img_2, target
        return img, target

    def __len__(self):
        return len(self.dataset)
# ========================================================
