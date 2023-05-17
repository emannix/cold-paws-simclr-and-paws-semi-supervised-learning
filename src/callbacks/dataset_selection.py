from pytorch_lightning.callbacks import Callback
from pdb import set_trace as pb

from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
import os
import pandas as pd
import glob

from torchmetrics import AUROC, AveragePrecision
import torch
import numpy as np

# =======================================================================================================================

class LoadInitData(Callback):

    def setup(self, trainer, pl_module, stage=None):
        pass



