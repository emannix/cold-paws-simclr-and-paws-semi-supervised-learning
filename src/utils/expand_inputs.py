import torch

def expand_inputs_y_unsup(x, classes):
    ys = [y_i for y_i in range(classes)]
    ys = torch.tensor(ys).long().to(x.device)
    ys = ys.repeat(x.shape[0], 1).flatten()

def expand_inputs(x, classes):
    ys = [y_i for y_i in range(classes)]
    ys = torch.tensor(ys).long().to(x.device)
    ys = ys.repeat(x.shape[0], 1).flatten()
    xs = torch.repeat_interleave(x, classes, dim=0)
    return xs, ys