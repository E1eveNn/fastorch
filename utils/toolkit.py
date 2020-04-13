import numpy as np
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
import torch.optim as optim


def split_data(arrays, start=0, end=None):
    arrays = np.array(arrays)
    if isinstance(arrays, list):
        if end is None:
            return [x[start:] for x in arrays]
        else:
            return [x[start: end] for x in arrays]
    else:
        if end is None:
            return arrays[start:]
        else:
            return arrays[start: end]




def get_optimizer(optimizer, model):
    if isinstance(optimizer, str):
        optimizer = optimizer.lower()
        if optimizer in ['sgd']:
            return optim.SGD(model.parameters(), lr=1e-2)
        elif optimizer in ['adam']:
            return optim.Adam(model.parameters())
        else:
            raise ValueError('Unknwon optimizer type!')
    elif isinstance(optimizer, Optimizer):
        return optimizer



def get_objective(objective):
    if isinstance(objective, str):
        objective = objective.lower()
        if objective in ['l1', 'l1loss']:
            return nn.L1Loss()
        elif objective in ['nll', 'nllloss']:
            return nn.NLLLoss()
        elif objective in ['nll2d', 'nllloss2d']:
            return nn.NLLLoss2d()
        elif objective in ['poissonnll', 'poissonnllloss']:
            return nn.PoissonNLLLoss()
        elif objective in ['kldiv', 'kldivloss']:
            return nn.KLDivLoss()
        elif objective in ['mse', 'mseloss']:
            return nn.MSELoss()
        elif objective in ['bce', 'bceloss']:
            return nn.BCELoss()
        elif objective in ['smoothl1', 'smoothl1loss']:
            return nn.SmoothL1Loss()
        elif objective in ['crossentropy', 'cross_entropy']:
            return nn.CrossEntropyLoss()
        elif objective in ['ctc', 'ctcloss']:
            return nn.CTCLoss()
        else:
            raise ValueError('unknown argument!')
    elif isinstance(objective, _Loss):
        return objective
    else:
        raise ValueError('unknown argument {}'.format(objective))

