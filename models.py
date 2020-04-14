from typing import List, Union, Tuple
from torch.utils.data import Dataset, DataLoader
from numpy import ndarray
from torch.optim.optimizer import Optimizer
import torch
from torchvision.transforms import transforms
from .functional import fit, evaluate



class Model(object):
    def __init__(self):
        super(Model, self).__init__()
        self.history = None

    def fit(self,
            train_dataset: Dataset = None,
            x: Union[ndarray, List] = None,
            y: Union[ndarray, List] = None,
            optimizer: Union[Optimizer, str] = None,
            criterion: Union[torch.nn.Module, str] = None,
            transform: transforms = None,
            batch_size: int = None,
            epochs: int = 1,
            verbose: int = 1,
            print_acc: bool = True,
            callbacks: List = None,
            validation_dataset: Dataset = None,
            validation_split: float = 0.0,
            validation_data: Union[torch.Tensor, ndarray, List] = None,
            validation_transform: transforms = None,
            shuffle: bool = True,
            initial_epoch: int = 0,
            steps_per_epoch: int = None,
            device: str = None,
            **kwargs) -> dict:

        history = fit(self,
                   train_dataset,
                   x,
                   y,
                   optimizer,
                   criterion,
                   transform,
                   batch_size,
                   epochs,
                   verbose,
                   print_acc,
                   callbacks,
                   validation_dataset,
                   validation_split,
                   validation_data,
                   validation_transform,
                   shuffle,
                   initial_epoch,
                   steps_per_epoch,
                   device,
                   **kwargs)
        self.history = history
        return history


    def evaluate(self,
                 dataset: Dataset = None,
                 dataloader: DataLoader = None,
                 x: Union[ndarray, List] = None,
                 y: Union[ndarray, List] = None,
                 transform: transforms = None,
                 batch_size: int = None,
                 verbose: int = 1,
                 criterion: Union[torch.nn.Module, str] = None,
                 print_acc: bool = True,
                 steps: int = None,
                 device: str = None,
                 **kwargs) -> Tuple:
        return evaluate(self, dataset, dataloader, x, y, transform, batch_size, verbose, criterion, print_acc, steps,
                        device, **kwargs)
