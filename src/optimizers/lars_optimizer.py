import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from pdb import set_trace as pb 

class LARS(Optimizer):
    """
    Slight modification of LARC optimizer from https://github.com/NVIDIA/apex/blob/d74fda260c403f775817470d87f810f816f3d615/apex/parallel/LARC.py
    Matches one from SimCLR implementation https://github.com/google-research/simclr/blob/master/lars_optimizer.py

    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the adaptive lr. See https://arxiv.org/abs/1708.03888
    """

    def __init__(self,
                 optimizer,
                 trust_coefficient=0.001,
                 division_by_zero_thresh=1e-10
                 ):
        self.param_groups = optimizer.param_groups
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.state = self.optim.state
        self.defaults = self.optim.defaults
        self.division_by_zero_thresh = division_by_zero_thresh

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    def __repr__(self):
        return self.optim.__repr__()

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue

                    if weight_decay != 0:
                        p.grad.data += weight_decay * p.data

                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)
                    adaptive_lr = 1.

                    if param_norm > self.division_by_zero_thresh and grad_norm > self.division_by_zero_thresh and group['layer_adaptation']:
                        adaptive_lr = self.trust_coefficient * param_norm / grad_norm
                    # pb()
                    p.grad.data *= adaptive_lr

        
        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]
        return loss

# https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/optimizers/lars.py

"""
References:
    - https://arxiv.org/pdf/1708.03888.pdf
    - https://github.com/pytorch/pytorch/blob/1.6/torch/optim/sgd.py
"""
# import torch
# from torch.optim.optimizer import Optimizer, required
# from pdb import set_trace as pb 

# class LARS(Optimizer):
#     """Extends SGD in PyTorch with LARS scaling from the paper
#     `Large batch training of Convolutional Networks <https://arxiv.org/pdf/1708.03888.pdf>`_.
#     Args:
#         params (iterable): iterable of parameters to optimize or dicts defining
#             parameter groups
#         lr (float): learning rate
#         momentum (float, optional): momentum factor (default: 0)
#         weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#         dampening (float, optional): dampening for momentum (default: 0)
#         nesterov (bool, optional): enables Nesterov momentum (default: False)
#         trust_coefficient (float, optional): trust coefficient for computing LR (default: 0.001)
#         eps (float, optional): eps for division denominator (default: 1e-8)
#     Example:
#         >>> model = torch.nn.Linear(10, 1)
#         >>> input = torch.Tensor(10)
#         >>> target = torch.Tensor([1.])
#         >>> loss_fn = lambda input, target: (input - target) ** 2
#         >>> #
#         >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
#         >>> optimizer.zero_grad()
#         >>> loss_fn(model(input), target).backward()
#         >>> optimizer.step()
#     .. note::
#         The application of momentum in the SGD part is modified according to
#         the PyTorch standards. LARS scaling fits into the equation in the
#         following fashion.
#         .. math::
#             \begin{aligned}
#                 g_{t+1} & = \text{lars_lr} * (\beta * p_{t} + g_{t+1}), \\
#                 v_{t+1} & = \\mu * v_{t} + g_{t+1}, \\
#                 p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
#             \\end{aligned}
#         where :math:`p`, :math:`g`, :math:`v`, :math:`\\mu` and :math:`\beta` denote the
#         parameters, gradient, velocity, momentum, and weight decay respectively.
#         The :math:`lars_lr` is defined by Eq. 6 in the paper.
#         The Nesterov version is analogously modified.
#     .. warning::
#         Parameters with weight decay set to 0 will automatically be excluded from
#         layer-wise LR scaling. This is to ensure consistency with papers like SimCLR
#         and BYOL.
#     """

#     def __init__(
#         self,
#         params,
#         lr=required,
#         momentum=0,
#         dampening=0,
#         weight_decay=0,
#         nesterov=False,
#         trust_coefficient=0.001,
#         eps=1e-8,
#     ):
#         if lr is not required and lr < 0.0:
#             raise ValueError(f"Invalid learning rate: {lr}")
#         if momentum < 0.0:
#             raise ValueError(f"Invalid momentum value: {momentum}")
#         if weight_decay < 0.0:
#             raise ValueError(f"Invalid weight_decay value: {weight_decay}")

#         defaults = dict(
#             lr=lr,
#             momentum=momentum,
#             dampening=dampening,
#             weight_decay=weight_decay,
#             nesterov=nesterov,
#             trust_coefficient=trust_coefficient,
#             eps=eps,
#         )
#         if nesterov and (momentum <= 0 or dampening != 0):
#             raise ValueError("Nesterov momentum requires a momentum and zero dampening")

#         super().__init__(params, defaults)

#     def __setstate__(self, state):
#         super().__setstate__(state)

#         for group in self.param_groups:
#             group.setdefault("nesterov", False)

#     @torch.no_grad()
#     def step(self, closure=None):
#         """Performs a single optimization step.
#         Args:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         # exclude scaling for params with 0 weight decay
#         for group in self.param_groups:
#             weight_decay = group["weight_decay"]
#             momentum = group["momentum"]
#             dampening = group["dampening"]
#             nesterov = group["nesterov"]

#             for p in group["params"]:
#                 if p.grad is None:
#                     continue

#                 d_p = p.grad
#                 p_norm = torch.norm(p.data)
#                 g_norm = torch.norm(p.grad.data)

#                 # lars scaling + weight decay part
#                 if weight_decay != 0:
#                     if p_norm != 0 and g_norm != 0:
#                         lars_lr = p_norm / (g_norm + p_norm * weight_decay + group["eps"])
#                         lars_lr *= group["trust_coefficient"]

#                         d_p = d_p.add(p, alpha=weight_decay)
#                         d_p *= lars_lr

#                 # sgd part
#                 if momentum != 0:
#                     param_state = self.state[p]
#                     if "momentum_buffer" not in param_state:
#                         buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
#                     else:
#                         buf = param_state["momentum_buffer"]
#                         buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
#                     if nesterov:
#                         d_p = d_p.add(buf, alpha=momentum)
#                     else:
#                         d_p = buf

#                 p.add_(d_p, alpha=-group["lr"])
#         pb()
#         return loss