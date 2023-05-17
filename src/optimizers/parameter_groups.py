
import torch
from pdb import set_trace as pb

def set_parameter_groups(self):
    if (hasattr(self, 'parameter_groups_wd_exclude')):
        param_wd_exclude = self.parameter_groups_wd_exclude # 'bn', 'bias', 'discriminator'
        param_lars_exclude = self.parameter_groups_lars_exclude # 'bn', 'bias', 'discriminator'
        grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if (not any(nd in n for nd in param_wd_exclude) and not any(nd in n for nd in param_lars_exclude))], 
            'weight_decay': self.weight_decay, 'layer_adaptation': True, 'LARS_exclude': False},
            {'params': [p for n, p in self.named_parameters() if  (not any(nd in n for nd in param_wd_exclude) and any(nd in n for nd in param_lars_exclude))], 
            'weight_decay': self.weight_decay, 'layer_adaptation': False, 'LARS_exclude': True},
            {'params': [p for n, p in self.named_parameters() if  (any(nd in n for nd in param_wd_exclude) and not any(nd in n for nd in param_lars_exclude))], 
            'weight_decay': 0.0, 'layer_adaptation': True, 'LARS_exclude': False},
            {'params': [p for n, p in self.named_parameters() if (any(nd in n for nd in param_wd_exclude) and any(nd in n for nd in param_lars_exclude))], 
            'weight_decay': 0.0, 'layer_adaptation': False, 'LARS_exclude': True}
        ]
    else:
        grouped_parameters = self.parameters()
    # [n for n, p in self.named_parameters()]
    return grouped_parameters

