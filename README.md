# Fastorch

[Pytorch](https://github.com/pytorch/pytorch) is convenient and easy to use, while [Keras](https://github.com/keras-team/keras) is designed to experiment quickly. Do you want to possess **both** Pytorch's convenience and Keras'  fast experimentation? The answer is *fastorch.*



## Traditional classification task training flow in pytorch

```python
################################# import dependencies ###############################
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

################################# define your network ###############################
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 10),
        )
        
    def forward(self, x):
        return self.fc(x)
   
################################## prepare dataset ##################################
class mydataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, item):
        return self.x[item], self.y[item]
    
    def __len__(self):
        return self.x.size(0)
# for simplicity, randomly generate datas, 
X = torch.randn(5000, 1000)
Y = torch.randint(0, 10, size=(5000,))
BATCH_SIZE = 100
trainset = mydataset(X, Y)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

########################### assign optimizer and criterion ##########################
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

############################## define and run training ##############################
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCH = 10
net = network().to(device)
for epoch in range(EPOCH):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = net(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print('Epoch %d -> loss: %.4f' % (epoch, loss.item()))
    
```

It's simple and straightforward, but is that enough? Below gives **fastorch resolution.**



## Training with fastorch

```python
################################# import dependencies ###############################
import torch.nn as nn
import fastorch

################################# define your network ###############################
'''
inherit from fastorch in your own network
'''
class network(fastorch.models.Model, nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 10),
        )
        
    def forward(self, x):
        return self.fc(x)

################################## prepare dataset ##################################
# for simplicity, randomly generate datas, 
X = torch.randn(5000, 1000)
Y = torch.randint(0, 10, size=(5000,))

#################################### run training ###################################
net = network()
net.fit(x=X, y=Y, optimizer='sgd', criterion='cross_entropy', batch_size=BATCH_SIZE, epochs=1, verbose=1)
'''
Epoch[1/1]
 2100/5000 [========>***********] - 1s - 68ms/batch -batch_loss: 2.2993 -batch_acc: 0.1200
'''
```



## Installation

Before installing Fastorch, please install the following **dependencies**:

- python >= 3.0 (3.6 is recommend)
- pytorch (1.0+ is recommend)

Then you can install Fastorchby using pip:

```
$ pip install fastorch
```



## Supports

When using `fastorch.models.Model`, make sure your network is **inherited** from both `fastorch.models.Model` and `torch.nn.Module`.

### - fastorch.models.Model

**fit**(*train_dataset*=None, *x*=None, *y*=None, *optimizer*=None, *criterion*=None, *transform*=None, *batch_size*=None, *epochs*=1, *verbose*=1, *print_acc*=True, *callbacks*=None, *validation_dataset*=None, *validation_split*=0.0, *validation_data*=None, *validation_transform*=None, *shuffle*=True, *initial_epoch*=0, *steps_per_epoch*=None, *device*=None, **kwargs)

+ *train_dataset*: training dataset, which inherited from `torch.utils.data.Dataset`. Make sure *x* and *y* are not None when *train_dataset* is None.
+ *x*: training data, it can be `Tensor`、`ndarray` or `List`. You need pass *x* and *y* when *train_dataset* is None.
+ *y*: training target, it can be `Tensor`、`ndarray` or `List`. You need pass *x* and *y* when *train_dataset* is None.
+ *optimizer*: model optimizer, it can be `str` or an instance of `pytorch optimizer`, when passing str('sgd', 'adam'), fastorch will automatically declare the corresponding pytorch optimizer.
+ *criterion*: loss function, it can be `str` or an instance of `pytorch objective function  `, when passing str('mse', 'cross_entropy', etc.), fastorch will automatically declare the corresponding pytorch objective function .
+ transform:  applying `torchvision.transforms` on *x* and *y* or *train_dataset*.
+ *batch_size*: training batch_size,  default is 32 if not specified.
+ *epochs*: training stop epochs, default is 1 if not specified.
+ *verbose*: log display mode, 0 = silent, 1 = progress bar, 2 = print each line.
+ *print_acc*: True or False, passing True if you are doing classification task.
+ *callbacks*: waiting for implemented.
+ *validation_dataset*:  validation dataset, which inherited from `torch.utils.data.Dataset`. 
+ *validation_split*: split validation data from training data's ratio, make sure *x* and *y* are not None when using this argument.
+ *validation_data*: tuple of validation x and validation y.
+ *validation_transform*:  applying `torchvision.transforms` on *validation_data* or data split by *validation_split*.
+ *shuffle*: whether shuffle the training data.
+ *initial_epoch*: start training epoch.
+ *steps_per_epoch*: int or None, if specified, batch_size will be set as total_sample_nums / *steps_per_epoch*.
+ *device*: specify device, otherwise fastorch will set device according to `torch.cuda.is_available()`.
+ **kwargs: arguments used for `torch.utils.data.Dataloader`, etc.



**evaluate**(*dataset*=None, *dataloader*=None, *x*=None, *y*=None, *transform*=None, *batch_size*=None, *verbose*=1, *criterion*=None, *print_acc*=True, *steps*=None, *device*=None, **kwargs)

+ *dataset*: evaluate dataset, which inherited from `torch.utils.data.Dataset`. Make sure *dataloader* or  *x* and *y* are not None when *dataset*is None.

+ *dataloader*: evaluate dataloader, which inherited from `torch.utils.data.Dataloader`. Make sure *dataset*or  *x* and *y* are not None when *dataloader*is None.

+ *x*: test data, it can be `Tensor`、`ndarray` or `List`. You need pass *x* and *y* when *dataset* or *dataloader* is None.

+ *y*: test target, it can be `Tensor`、`ndarray` or `List`. You need pass *x* and *y* when *dataset* or *dataloader* is None.

+ transform:  applying `torchvision.transforms` on *x* and *y* or *dataset*.

+ *batch_size*: evaluate batch_size,  default is 32 if not specified.

+ *verbose*: log display mode, 0 = silent, 1 = progress bar, 2 = print each line.

+ *criterion*: loss function, it can be `str` or an instance of `pytorch objective function  `, when passing str('mse', 'cross_entropy', etc.), fastorch will automatically declare the corresponding pytorch objective function .

+ *print_acc*: True or False, passing True if you are doing classification task.

+ *steps*: int or None, if specified, batch_size will be set as total_sample_nums / *steps*.

+ *device*: specify device, otherwise fastorch will set device according to `torch.cuda.is_available()`.

+ **kwargs: arguments used for `torch.utils.data.Dataloader`, etc.

  

### - fastorch.functional

**fit**(*model*=None, *train_dataset*=None, *x*=None, *y*=None, *optimizer*=None, *criterion*=None, *transform*=None, *batch_size*=None, *epochs*=1, *verbose*=1, *print_acc*=True, *callbacks*=None, *validation_dataset*=None, *validation_split*=0.0, *validation_data*=None, *validation_transform*=None, *shuffle*=True, *initial_epoch*=0, *steps_per_epoch*=None, *device*=None, **kwargs)

+ *model*: your network, must be specify.

+ *train_dataset*: training dataset, which inherited from `torch.utils.data.Dataset`. Make sure *x* and *y* are not None when *train_dataset* is None.
+ *x*: training data, it can be `Tensor`、`ndarray` or `List`. You need pass *x* and *y* when *train_dataset* is None.
+ *y*: training target, it can be `Tensor`、`ndarray` or `List`. You need pass *x* and *y* when *train_dataset* is None.
+ *optimizer*: model optimizer, it can be `str` or an instance of `pytorch optimizer`, when passing str('sgd', 'adam'), fastorch will automatically declare the corresponding pytorch optimizer.
+ *criterion*: loss function, it can be `str` or an instance of `pytorch objective function  `, when passing str('mse', 'cross_entropy', etc.), fastorch will automatically declare the corresponding pytorch objective function .
+ transform:  applying `torchvision.transforms` on *x* and *y* or *train_dataset*.
+ *batch_size*: training batch_size,  default is 32 if not specified.
+ *epochs*: training stop epochs, default is 1 if not specified.
+ *verbose*: log display mode, 0 = silent, 1 = progress bar, 2 = print each line.
+ *print_acc*: True or False, passing True if you are doing classification task.
+ *callbacks*: waiting for implemented.
+ *validation_dataset*:  validation dataset, which inherited from `torch.utils.data.Dataset`. 
+ *validation_split*: split validation data from training data's ratio, make sure *x* and *y* are not None when using this argument.
+ *validation_data*: tuple of validation x and validation y.
+ *validation_transform*:  applying `torchvision.transforms` on *validation_data* or data split by *validation_split*.
+ *shuffle*: whether shuffle the training data.
+ *initial_epoch*: start training epoch.
+ *steps_per_epoch*: int or None, if specified, batch_size will be set as total_sample_nums / *steps_per_epoch*.
+ *device*: specify device, otherwise fastorch will set device according to `torch.cuda.is_available()`.
+ **kwargs: arguments used for `torch.utils.data.Dataloader`, etc.



**evaluate**((*model*=None, *dataset*=None, *dataloader*=None, *x*=None, *y*=None, *transform*=None, *batch_size*=None, *verbose*=1, *criterion*=None, *print_acc*=True, *steps*=None, *device*=None, **kwargs)

+ *model*: your network, must be specify.

+ *dataset*: evaluate dataset, which inherited from `torch.utils.data.Dataset`. Make sure *dataloader* or  *x* and *y* are not None when *dataset*is None.
+ *dataloader*: evaluate dataloader, which inherited from `torch.utils.data.Dataloader`. Make sure *dataset*or  *x* and *y* are not None when *dataloader*is None.
+ *x*: test data, it can be `Tensor`、`ndarray` or `List`. You need pass *x* and *y* when *dataset* or *dataloader* is None.
+ *y*: test target, it can be `Tensor`、`ndarray` or `List`. You need pass *x* and *y* when *dataset* or *dataloader* is None.
+ transform:  applying `torchvision.transforms` on *x* and *y* or *dataset*.
+ *batch_size*: evaluate batch_size,  default is 32 if not specified.
+ *verbose*: log display mode, 0 = silent, 1 = progress bar, 2 = print each line.
+ *criterion*: loss function, it can be `str` or an instance of `pytorch objective function  `, when passing str('mse', 'cross_entropy', etc.), fastorch will automatically declare the corresponding pytorch objective function .
+ *print_acc*: True or False, passing True if you are doing classification task.
+ *steps*: int or None, if specified, batch_size will be set as total_sample_nums / *steps*.
+ *device*: specify device, otherwise fastorch will set device according to `torch.cuda.is_available()`.
+ **kwargs: arguments used for `torch.utils.data.Dataloader`, etc.

