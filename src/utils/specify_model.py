import torch
from pdb import set_trace as pb

class SpecifyModel:
    def __init__(self, model_class, params):
        self.model_class = model_class
        self.params = params

    def init_model(self):
        return self.model_class(**self.params)

    def stringify(self):
        return {
        'object_class': self.model_class.__name__,
        'parameters': self.params
    }

    def __str__(self):
        return self.model_class.__name__ + str(self.params)

