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

import kornia as K
from kornia.augmentation import (
    CenterCrop,
    ColorJiggle,
    ColorJitter,
    PadTo,
    RandomAffine,
    RandomBoxBlur,
    RandomBrightness,
    RandomChannelShuffle,
    RandomContrast,
    RandomCrop,
    RandomCutMixV2,
    RandomElasticTransform,
    RandomEqualize,
    RandomErasing,
    RandomFisheye,
    RandomGamma,
    RandomGaussianBlur,
    RandomGaussianNoise,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomHue,
    RandomInvert,
    RandomJigsaw,
    RandomMixUpV2,
    RandomMosaic,
    RandomMotionBlur,
    RandomPerspective,
    RandomPlanckianJitter,
    RandomPlasmaBrightness,
    RandomPlasmaContrast,
    RandomPlasmaShadow,
    RandomPosterize,
    RandomResizedCrop,
    RandomRGBShift,
    RandomRotation,
    RandomSaturation,
    RandomSharpness,
    RandomSolarize,
    RandomThinPlateSpline,
    RandomVerticalFlip,
)

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


###################################################################
def show_example(img, label,dataset):
    print('Label: ', dataset.classes[label], "(" + str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))
    # convert torch.Size([3, 32, 32]) to torch.Size([32, 32, 3])

###################################################################
def display_images_for_label(dataset, label):
    # Collect indices of images for the specified label
    label_indices = [idx for idx, (_, lbl) in enumerate(dataset) if lbl == label]
    
    # Create a figure for displaying images
    num_images = 10
    num_cols = 10
    num_rows = 1

    print(dataset.classes[label])
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(11, 4))
    
    # Display images for the specified label
    for i, idx in enumerate(label_indices[:num_images]):
        img, label = dataset[idx]
        plt.subplot(1,10,i+1)
        plt.axis('off')
        plt.imshow(img.permute(1, 2, 0))
        
    plt.show()

###################################################################
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break

###################################################################   
def visual_kornia(aug_method,img_reference, img):
    plt.figure(figsize=(11, 4))
    plt.subplot(1,2,1)
    plt.imshow(img_reference)

    transformed_img = aug_method(img)
    plt.subplot(1,2,2)
    plt.imshow(transformed_img[0,:,:,:].permute(1, 2, 0))


