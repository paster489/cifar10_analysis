import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import json
import math
from argparse import ArgumentParser
import multiprocessing
import random

from utils_data import *

###################################################################
# Use CUDA Device
###################################################################
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
################################################################        
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
################################################################
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

###################################################################
# Definition of model per user input
###################################################################
def chose_model(hparams_model,device):
    if hparams_model == 'CNN':
        from models.CNN_model import Cifar10CnnModel
        model = to_device(Cifar10CnnModel(), device)
    elif hparams_model == 'ResNet_18':
        from models.ResNet_model import ResNet, BasicBlock
        model = to_device(ResNet(BasicBlock, [2, 2, 2, 2]), device)
    elif hparams_model == 'ResNet_34':
        from models.ResNet_model import ResNet, BasicBlock
        model = to_device(ResNet(BasicBlock, [3, 4, 6, 3]), device)
    elif hparams_model == 'ViT':
        from models.ViT_model import ViT
        model = to_device(ViT(
                                image_size = 32,
                                patch_size = 4,
                                num_classes = 10,
                                dim = 512,
                                depth = 6,
                                heads = 8,
                                mlp_dim = 512,
                                dropout = 0.1,
                                emb_dropout = 0.1
                            ), device)
    elif hparams_model == 'ViT_small':
        from models.ViT_small_model import ViT
        model = to_device(ViT(
                                image_size = 32,
                                patch_size = 4,
                                num_classes = 10,
                                dim = 512,
                                depth = 6,
                                heads = 8,
                                mlp_dim = 512,
                                dropout = 0.1,
                                emb_dropout = 0.1
                            ), device)
    elif hparams_model == 'ViT_tiny':
        from models.ViT_small_model import ViT
        model = to_device(ViT(
                                image_size = 32,
                                patch_size = 4,
                                num_classes = 10,
                                dim = 512,
                                depth = 4,
                                heads = 6,
                                mlp_dim = 512,
                                dropout = 0.1,
                                emb_dropout = 0.1
                            ), device)
    elif hparams_model == 'ViT_simple':
        from models.ViT_simple_model import SimpleViT
        model = to_device(SimpleViT(
                            image_size = 32,
                            patch_size = 4,
                            num_classes = 10,
                            dim = 512,
                            depth = 6,
                            heads = 8,
                            mlp_dim = 512
                        ), device)
    else:
        print('Model is not defined')
            
    return model

###################################################################
# Define type of preprocessing according to user input
###################################################################
def chose_preprocess(hparams_norm,data_dir):
    if hparams_norm == "No":
        train_ds, test_ds = simple_DL(data_dir)
    elif hparams_norm == "N1":
        train_ds, test_ds = norm_1_DL(data_dir)
    elif hparams_norm == "N2":
        train_ds, test_ds = norm_2_DL(data_dir)
    elif hparams_norm == "N1_aug":
        train_ds, test_ds = norm_1_aug_DL(data_dir)
    elif hparams_norm == "N2_aug":
        train_ds, test_ds = norm_2_aug_DL(data_dir)
    elif hparams_norm == "aug":
        train_ds, test_ds = aug_DL(data_dir)
    else:
        print('Error in normalization parameter input')

    return train_ds, test_ds

###################################################################
# Define optimizer according to user input
###################################################################
def chose_optimizer(hparams_opt):
    if hparams_opt == 'Adam':
        opt_func = torch.optim.Adam
    elif hparams_opt == 'SGD':
        opt_func = torch.optim.SGD
    else:
        print('Error in optimasation parameter input')

    return opt_func