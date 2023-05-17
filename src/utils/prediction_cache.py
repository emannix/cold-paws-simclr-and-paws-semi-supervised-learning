import torchmetrics
import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from pdb import set_trace as pb

class PredictionCache(Metric):
    full_state_update=True
    
    def __init__(self,
            # output_pred,
            dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
    
        # self.output_pred = output_pred
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        # self.add_state("preds_raw", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        # preds_raw = preds
        # preds = self.output_pred(preds)
        
        self.preds.append(preds.detach().cpu())
        # self.preds_raw.append(preds_raw)
        self.target.append(target.detach().cpu())

    def reset(self):
        self.target = []
        self.preds = []
        # self.preds_raw = []
        
    def return_results(self):
        # preds_raw = dim_zero_cat(self.preds_raw).squeeze()
        if (len(self.preds) > 0):
            preds = dim_zero_cat(self.preds).squeeze()
            target = dim_zero_cat(self.target).squeeze()
        else:
            preds, target = torch.tensor(0.0), torch.tensor(0.0)
        return(preds, target)
    
    def compute(self):
        pass