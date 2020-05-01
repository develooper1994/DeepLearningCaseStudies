# TODO! It is not done yet
# pytorch imports
import torch
import torchvision
from torchvision import datasets, models, transforms, utils
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn import functional, Sequential

# other 3rd party imports
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Standard Python modules
from collections import OrderedDict

# control the versions
print("Cuda version: ", torch.version.cuda)
print("Cudnn enabled?: ", torch.backends.cudnn.enabled)
print("Cudnn version: ", torch.backends.cudnn.version())
print("torch version: ", torch.__version__)
print("torchvision version: ", torchvision.__version__)
print("numpy version: ", np.__version__)
print("pd version: ", pd.__version__)

# check the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Tensor backend uses: ", device)


class TheDataset(datasets):
    """
    The dataset loader.
    Splits dataset into TRAIN, VALIDATION and TEST
    """
    pass


class TheModel(nn.Module):
    """
    The main model(or neural network) itself
    """
    def __init__(self):
        super().__init__()
        pass

    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        pass


class NetRunner:
    """
    NetRunner contains loss, optimizer functions and lr_scheduler
     NetRunner trains and test the model
     """
    pass
