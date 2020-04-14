import numpy as np
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
import torch.optim as optim


class ProgressBar(object):
    def __init__(self,
                 max_iter: int = 1,
                 verbose: int = 1,
                 bar_nums: int = 20,
                 untrained_sign: str = '*',
                 trained_sign: str = '='):
        self.max_iter = max_iter
        self.verbose = verbose
        self._nums = bar_nums - 1
        self._untrained = untrained_sign
        self._trained = trained_sign
        self.iter = 0

    def update(self, n_iter: int = 1):
        self.iter += n_iter

    def get_bar(self) -> str:
        trained_ratio = self.iter / self.max_iter
        reached_bar_nums = round(trained_ratio * self._nums)
        unreached_bar_nums = self._nums - reached_bar_nums
        if self.verbose == 1:
            bar = reached_bar_nums * self._trained + '>' + unreached_bar_nums * self._untrained
        else:
            percent = str(round(trained_ratio * 100))
            bar = '{black} {percent:>{white}}%'.format(black="\033[40m%s\033[0m" % ' ' * reached_bar_nums,
                                                       percent=percent, white=unreached_bar_nums)
        return bar



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


def console(prog_bar: ProgressBar = None,
            verbose: int = 0,
            trained_samples: int = None,
            total_samples: int = None,
            trained_batch: int = 1,
            total_batch: int = 1,
            trained_time: float = 0.,
            batch_loss: float = 0.,
            batch_acc: float = 0.,
            validation_loss: float = None,
            validation_acc: float = None):

    if verbose == 0:
        return
    elif verbose == 1:
        formated_trained_time = format_time(trained_time)
        formated_per_batch_time = format_time(trained_time / trained_batch)
        bar = prog_bar.get_bar()
        if validation_loss is None and validation_acc is None:
            print('\r {:d}/{:d} [{}] - {} - {}/batch -batch_loss: {:.4f} -batch_acc: {:.4f}'.format(trained_samples,
                                                                                                    total_samples, bar,
                                                                                                    formated_trained_time,
                                                                                                    formated_per_batch_time,
                                                                                                    batch_loss,
                                                                                                    batch_acc),
                  flush=True, end='')
        else:
            print('\r {:d}/{:d} [{}] - {} - {}/batch'
                  ' -batch_loss: {:.4f} -batch_acc: {:.4f} -validation_loss: {:.4f} -validation_acc: {:.4f}'.format(
                    trained_samples, total_samples, bar, formated_trained_time, formated_per_batch_time, batch_loss,
                    batch_acc, validation_loss, validation_acc), flush=True, end='')
    elif verbose == 2:
        batch_time = trained_time / trained_batch
        eta = (total_batch - trained_batch) * batch_time
        formated_eta = format_time(eta)
        bar = prog_bar.get_bar()
        if validation_loss is None and validation_acc is None:
            print('{} -ETA {} -batch_loss: {:.4f} -batch_acc: {:.4f}'.format(bar, formated_eta, batch_loss, batch_acc))
        else:
            print(
                '{} -ETA {} -batch_loss: {:.4f} -batch_acc: {:.4f} -validation_loss: {:.4f} -validation_acc: {:.4f}'.format(
                    bar, formated_eta, batch_loss, batch_acc, validation_loss, validation_acc))
    else:
        raise ValueError('Verbose only supports for 0, 1 and 2 ~')


def format_time(second_time: float) -> str:
    if second_time < 1:
        ms = second_time * 1000
        if ms < 1:
            us = second_time * 1000
            return '%dus' % us
        else:
            return '%dms' % ms
    second_time = round(second_time)
    if second_time > 3600:
        # hours
        h = second_time // 3600
        second_time = second_time % 3600
        # minutes
        m = second_time // 60
        second_time = second_time % 60
        return '%dh%dm%ds' % (h, m, second_time)
    elif second_time > 60:
        m = second_time // 60
        second_time = second_time % 60
        return '%dm%ds' % (m, second_time)
    else:
        return '%ds' % second_time

