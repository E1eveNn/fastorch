from typing import List, Union
from torch.utils.data import Dataset, DataLoader
from numpy import ndarray
from torch.optim.optimizer import Optimizer
from torch.nn import Module
from torch import Tensor
from torchvision.transforms import transforms
from .utils.data import new_dataset
from .utils.toolkit import *
import torch
import warnings
import time


class Model(object):
    def __init__(self):
        super(Model, self).__init__()
        self._bar_nums = 20
        self._bar_untrained = '*'
        self._bar_trained = '='


    def fit(self,
            train_dataset: Dataset = None,
            x: Union[ndarray, List] = None,
            y: Union[ndarray, List] = None,
            optimizer: Union[Optimizer, str] = None,
            criterion: Union[Module, str] = None,
            transform: transforms = None,
            batch_size: int = None,
            epochs: int = 1,
            verbose: int = 1,
            print_acc: bool = True,
            callbacks: List = None,
            validation_dataset: Dataset = None,
            validation_split: float = 0.0,
            validation_data: Union[Tensor, ndarray, List] = None,
            validation_transform: transforms = None,
            shuffle: bool = True,
            initial_epoch: int = 0,
            steps_per_epoch: int = None,
            device: str = None,
            **kwargs):

        if train_dataset is None and x is None and y is None:
            raise ValueError('You should specify the `dataset` or `x` and `y` argument')

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        optimizer = get_optimizer(optimizer, self)
        criterion = get_objective(criterion)

        if batch_size is None:
            if steps_per_epoch is None:
                batch_size = 32
            else:
                batch_size = len(train_dataset) // steps_per_epoch


        do_validation = False
        if validation_dataset is None:
            if validation_data:
                do_validation = True
                val_x, val_y = validation_data
                validation_dataset = new_dataset(val_x, val_y, validation_transform)
                validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, **kwargs)
            elif 0. < validation_split < 1. and validation_dataset is None:
                do_validation = True
                if hasattr(x[0], 'shape'):
                    split_at = int(x[0].shape[0] * (1 - validation_split))
                else:
                    split_at = int(len(x[0]) * (1 - validation_split))
                x, val_x = split_data(x, 0, split_at), split_data(x, split_at)
                y, val_y = split_data(y, 0, split_at), split_data(y, split_at)
                validation_dataset = new_dataset(val_x, val_y, validation_transform)
                validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        else:
            do_validation = True
            validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        if train_dataset and x and y:
            warnings.warn('`dataset`, `x`, and `y` arguments all are not None, however fastorch will use dataset only!')
        elif train_dataset is None:
            train_dataset = new_dataset(x, y, transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

        for epoch in range(initial_epoch, epochs):
            start_time = time.time()
            trained_samples = 0
            if verbose == 1:
                print('\033[1;31m Epoch[%d/%d]\033[0m' % (epoch + 1, epochs))
            for idx, (inputs, targets) in enumerate(train_loader):
                self.train()
                inputs, targets = inputs.to(device), targets.to(device)
                trained_samples += inputs.size(0)

                optimizer.zero_grad()
                out = self.forward(inputs)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()

                batch_end_time = time.time()
                batch_loss = loss.item()
                if print_acc:
                    pred = torch.max(out, 1)[1]
                    batch_acc = pred.eq(targets).sum().item() / pred.size(0)
                else:
                    batch_acc = 0.0

                if do_validation:
                    valid_loss, valid_acc = self.evaluate(dataset=validation_dataset, dataloader=validation_loader,
                                                          batch_size=batch_size, verbose=0,
                                                          criterion=criterion, print_acc=print_acc, device=device)
                else:
                    valid_loss = None
                    valid_acc = None
                self._console(verbose, trained_samples, len(train_dataset), idx + 1, len(train_loader),
                              batch_end_time - start_time, batch_loss, batch_acc, valid_loss, valid_acc)
            print()

    def evaluate(self,
                 dataset: Dataset = None,
                 dataloader: DataLoader = None,
                 x: Union[ndarray, List] = None,
                 y: Union[ndarray, List] = None,
                 transform: transforms = None,
                 batch_size: int = None,
                 verbose: int = 1,
                 criterion: Union[Module, str] = None,
                 print_acc: bool = True,
                 steps: int = None,
                 device: str = None,
                 **kwargs):
        if dataset is None and dataloader is None and x is None and y is None:
            raise ValueError('You should specify the `dataset` or `dataloader` or `x` and `y` argument')

        if batch_size is None and steps is None:
            batch_size = 32
        elif steps is not None:
            assert dataset is not None
            batch_size = len(dataset) // steps


        if dataset and dataloader and x and y:
            warnings.warn('`dataset`, `dataloader`,'
                          ' `x`, and `y` arguments all are not None, however fastorch will use dataloader only!')
        elif dataloader is None:
            if dataset is None:
                assert x and y, 'dataset and dataloader is None, make sure x and y are not None!'
                dataset = new_dataset(x, y, transform)

            dataloader = DataLoader(dataset, batch_size, shuffle=False, **kwargs)

        criterion = get_objective(criterion)

        if device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.to(device)
        self.eval()
        test_loss = 0.
        test_correct = 0.
        test_total = 0.
        test_acc = 0.
        with torch.no_grad():
            start_time = time.time()
            trained_samples = 0
            for idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                trained_samples += inputs.size(0)
                out = self.forward(inputs)

                if criterion:
                    loss = criterion(out, targets)
                    test_loss += loss.item()

                if print_acc:
                    pred = torch.max(out, 1)[1]
                    test_correct += pred.eq(targets).sum().item()
                    test_total += pred.size(0)
                    test_acc = test_correct / test_total

                batch_end_time = time.time()
                self._console(verbose, trained_samples, len(dataset), idx + 1, len(dataloader), batch_end_time - start_time, test_loss / (idx + 1), test_acc)
        return test_loss / len(dataloader), test_acc


    def _console(self, verbose, trained_samples=None, total_samples=None, trained_batch=1., total_batch=1.,trained_time=0., batch_loss=0., batch_acc=0., validation_loss=None, validation_acc=None):
        if verbose == 0:
            return
        elif verbose == 1:
            formated_trained_time = self._format_time(trained_time)
            formated_batch_time = self._format_time(trained_time / trained_batch)
            bar = self._progress_bar(trained_samples, total_samples, verbose)
            if validation_loss is None and validation_acc is None:
                print('\r {:d}/{:d} [{}] - {} - {}/batch -batch_loss: {:.4f} -batch_acc: {:.4f}'.format(trained_samples, total_samples, bar, formated_trained_time, formated_batch_time, batch_loss, batch_acc), flush=True, end='')
            else:
                print('\r {:d}/{:d} [{}] - {} - {}/batch -batch_loss: {:.4f} -batch_acc: {:.4f} -validation_loss: {:.4f} -validation_acc: {:.4f}'.format(
                    trained_samples, total_samples, bar, formated_trained_time, formated_batch_time, batch_loss,
                    batch_acc, validation_loss, validation_acc), flush=True, end='')
        elif verbose == 2:
            batch_time = trained_time / trained_batch
            eta = (total_batch - trained_batch) * batch_time
            formated_eta = self._format_time(eta)
            bar = self._progress_bar(trained_samples, total_samples, verbose)
            if validation_loss is None and validation_acc is None:
                print('{} -ETA {} -batch_loss: {:.4f} -batch_acc: {:.4f}'.format(bar, formated_eta, batch_loss, batch_acc))
            else:
                print('{} -ETA {} -batch_loss: {:.4f} -batch_acc: {:.4f} -validation_loss: {:.4f} -validation_acc: {:.4f}'.format(bar, formated_eta, batch_loss, batch_acc, validation_loss, validation_acc))
        else:
            raise ValueError('Verbose only supports for 0, 1 and 2 ~')


    def _progress_bar(self, trained_samples, total_samples, verbose=1):
        trained_ratio = trained_samples / total_samples
        trained_nums = round(trained_ratio * self._bar_nums)
        untrained_nums = self._bar_nums - trained_nums
        if verbose == 1:
            bar = self._bar_trained * trained_nums + '>' + self._bar_untrained * (untrained_nums - 1)
        else:
            percent = str(round(trained_samples * 100 / total_samples))
            bar = '{black} {percent:>{white}}%'.format(black="\033[40m%s\033[0m" % ' ' * trained_nums, percent=percent, white=untrained_nums)

        return bar


    def _format_time(self, second_time):
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







