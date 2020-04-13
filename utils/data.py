from torch.utils.data import Dataset
from typing import Union, List, Tuple
from numpy import ndarray
import numpy as np
from torchvision.transforms import transforms


class new_dataset(Dataset):
    def __init__(self,
                 x: Union[ndarray, List, Tuple, float],
                 y: Union[ndarray, List, Tuple, float] = None,
                 transform: transforms = None
                 ):
        self.x = np.array(x)
        self.y = np.array(y)
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        if self.transform is not None:
            x = self.transform(x)
        return x, y




