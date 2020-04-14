from typing import List, Union
from torch.utils.data import Dataset, DataLoader
from numpy import ndarray
from torch.optim.optimizer import Optimizer
from torch.nn import Module
from torchvision.transforms import transforms
from .utils.data import new_dataset
from .utils.toolkit import *
import torch
import warnings
import time


def fit(model: Module = None,
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
        validation_data:  List[ndarray] = None,
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
    model.to(device)

    optimizer = get_optimizer(optimizer, model)
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

    history = dict()
    for epoch in range(initial_epoch, epochs):
        history[epoch] = {}
        start_time = time.time()
        trained_samples = 0
        if verbose != 0:
            print('\033[1;31m Epoch[%d/%d]\033[0m' % (epoch + 1, epochs))
            prog_bar = ProgressBar(max_iter=len(train_loader), verbose=verbose)

        for idx, (inputs, targets) in enumerate(train_loader):
            model.train()
            inputs, targets = inputs.to(device), targets.to(device)
            trained_samples += inputs.size(0)

            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

            batch_end_time = time.time()
            batch_loss = loss.item()
            history[epoch].setdefault('train_loss', [batch_loss, ]).append(batch_loss)
            if print_acc:
                pred = torch.max(out, 1)[1]
                batch_acc = pred.eq(targets).sum().item() / pred.size(0)
                history[epoch].setdefault('train_acc', [batch_acc, ]).append(batch_acc)
            else:
                batch_acc = 0.0
            if do_validation:
                valid_loss, valid_acc = evaluate(model=model, dataset=validation_dataset, dataloader=validation_loader,
                                                      batch_size=batch_size, verbose=0,
                                                      criterion=criterion, print_acc=print_acc, device=device)
                history[epoch].setdefault('validation_loss', [valid_loss, ]).append(valid_loss)
                history[epoch].setdefault('validation_acc', [valid_acc, ]).append(valid_acc)
            else:
                valid_loss = None
                valid_acc = None

            prog_bar.update()
            console(prog_bar, verbose, trained_samples, len(train_dataset), idx + 1, len(train_loader),
                          batch_end_time - start_time, batch_loss, batch_acc, valid_loss, valid_acc)
        print()
    return history


def evaluate(model: Module = None,
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

        model.to(device)
        model.eval()
        test_loss = 0.
        test_correct = 0.
        test_total = 0.
        test_acc = 0.
        if verbose != 0:
            prog_bar = ProgressBar(max_iter=len(dataloader), verbose=verbose)
        with torch.no_grad():
            start_time = time.time()
            trained_samples = 0
            for idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                trained_samples += inputs.size(0)
                out = model(inputs)

                if criterion:
                    loss = criterion(out, targets)
                    test_loss += loss.item()

                if print_acc:
                    pred = torch.max(out, 1)[1]
                    test_correct += pred.eq(targets).sum().item()
                    test_total += pred.size(0)
                    test_acc = test_correct / test_total

                batch_end_time = time.time()
                if verbose != 0:
                    prog_bar.update()
                    console(prog_bar, verbose, trained_samples, len(dataset), idx + 1, len(dataloader),
                        batch_end_time - start_time,
                        test_loss / (idx + 1), test_acc)
        return test_loss / len(dataloader), test_acc
