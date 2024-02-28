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

###################################################################
# Datasets Load - Simple
###################################################################
def simple_DL(data_dir):   
    train_ds = torchvision.datasets.CIFAR10(data_dir+'/train',
                                            transform=Compose([transforms.ToTensor()]),
                                            train=True, download=True)
    
    test_ds = torchvision.datasets.CIFAR10(data_dir+'/test', 
                                           transform=Compose([transforms.ToTensor()]),
                                           train=False, download=True)
    return train_ds, test_ds

###################################################################
# Normalization 1
###################################################################
def norm_1_DL(data_dir):
    train_ds, test_ds = simple_DL(data_dir)
    
    norm_mean = np.round(train_ds.data.mean(axis=(0, 1, 2)) / 255, 4).tolist() 
    norm_std = np.round(train_ds.data.std(axis=(0, 1, 2)) / 255, 4).tolist()
    
    train_ds_norm_1 = torchvision.datasets.CIFAR10(data_dir+'/train',
                                            transform=Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=norm_mean, std=norm_std)
                                            ]),
                                            train=True, download=True)
    test_ds_norm_1 = torchvision.datasets.CIFAR10(data_dir+'/test', 
                                           transform=Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=norm_mean, std=norm_std)
                                           ]),
                                           train=False, download=True)
    return train_ds_norm_1, test_ds_norm_1

###################################################################
# Normalization 2
###################################################################
def norm_2_DL(data_dir):
    train_ds_norm_2 = torchvision.datasets.CIFAR10(data_dir+'/train',
                                            transform=Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize([0, 0, 0], [1, 1, 1])
                                            ]),
                                            train=True, download=True)
    test_ds_norm_2 = torchvision.datasets.CIFAR10(data_dir+'/test', 
                                           transform=Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize([0, 0, 0], [1, 1, 1])
                                           ]),
                                           train=False, download=True)


    return train_ds_norm_2, test_ds_norm_2

###################################################################
# Normalization 1 + Augmanation
###################################################################
def norm_1_aug_DL(data_dir):
    train_ds, test_ds = simple_DL(data_dir)
    
    norm_mean = np.round(train_ds.data.mean(axis=(0, 1, 2)) / 255, 4).tolist() 
    norm_std = np.round(train_ds.data.std(axis=(0, 1, 2)) / 255, 4).tolist()

    train_ds_norm_1_aug = torchvision.datasets.CIFAR10(data_dir+'/train',
                                            transform=Compose([
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=norm_mean, std=norm_std)
                                            ]),
                                            train=True, download=True)

    train_ds_norm_1, test_ds_norm_1 = norm_1_DL(data_dir)

    
    return train_ds_norm_1_aug, test_ds_norm_1

###################################################################
# Normalization 2 + Augmanation
###################################################################
def norm_2_aug_DL(data_dir):
    train_ds_norm_2, test_ds_norm_2 = norm_2_DL(data_dir)
    train_ds_norm_2_aug = torchvision.datasets.CIFAR10(data_dir+'/train',
                                            transform=Compose([
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0, 0, 0], [1, 1, 1])
                                            ]),
                                            train=True, download=True)
    return train_ds_norm_2_aug, test_ds_norm_2


###################################################################
# Augmanation
###################################################################
def aug_DL(data_dir):

    train_ds, test_ds = simple_DL(data_dir)

    train_ds_aug = torchvision.datasets.CIFAR10(data_dir+'/train',
                                            transform=Compose([
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor()
                                            ]),
                                            train=True, download=True)
    return train_ds_aug, test_ds


###################################################################
# DataLoader
###################################################################
def dataloader_help(train_ds, val_ds,n_workers,batch_size,g,seed_worker):
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=n_workers, pin_memory=True, generator=g, worker_init_fn=seed_worker)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=n_workers, pin_memory=True, generator=g, worker_init_fn=seed_worker)
        
    return train_dl, val_dl




