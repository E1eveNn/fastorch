import torch.utils.data
from typing import Union, List, Tuple
import torch
import numpy
import torchvision


class new_dataset(torch.utils.data.Dataset):
    def __init__(self,
                 x: Union[torch.Tensor, numpy.ndarray,List, Tuple, float],
                 y: Union[torch.Tensor, numpy.ndarray, List, Tuple, float] = None,
                 transform: torchvision.transforms = None
                 ):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        if hasattr(self.x, 'shape'):
            return self.x.shape[0]
        elif hasattr(self.x, 'size'):
            return self.x.size(0)
        else:
            return len(self.x)

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        if self.transform is not None:
            x = self.transform(x)
        return x, y
