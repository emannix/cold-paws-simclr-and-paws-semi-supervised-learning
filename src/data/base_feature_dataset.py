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

class BaseFeatureCSVDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, transform=None,
        representation_csv_file='test.csv', label_csv_file = 'test.csv', 
        label_csv_column = 'label', 
        return_index = True):
        'Initialization'
        
        label_sheet = pd.read_csv(label_csv_file)
        labels = label_sheet[label_csv_column].values

        rep_sheet = pd.read_csv(representation_csv_file)
        imgs = torch.tensor(rep_sheet.to_numpy()).float()

        self.transform = transform
        self.data = imgs
        self.targets = torch.tensor(labels)
        self.return_index = return_index
 
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        target = self.targets[index]
        image = self.data[index, :]
        if self.transform is not None:
            image = self.transform(image)
        if (self.return_index):
          return image, target, index
        else:
          return image, target # just grab the image, don't get anything else

