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
import time
import datetime

from utils_data import *
from utils_train import *
from utils_test import *
from utils_hparams import *

import matplotlib
matplotlib.use('Agg')  # Use Agg backend

import warnings
from sklearn.exceptions import UndefinedMetricWarning
# To ignore the UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

###################################################################
# Limiting randomness => https://pytorch.org/docs/stable/notes/randomness.html
###################################################################
random_seed = 42 # for custom operators
torch.manual_seed(random_seed) # for torch
np.random.seed(random_seed) # for NumPy libraries
    
# Use deterministic algorithms only
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
    
# Use deterministric convolution algorithm in CUDA
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
    
# Fix workers randomness
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
        
g = torch.Generator()
g.manual_seed(random_seed)

###################################################################
# MAIN
###################################################################
def main(hparams):
    print(' ')
    print('-------------------------------------------------------------------------------')
    print('main => start')
    print(' ')

    ###################################################################
    # Paths
    ###################################################################
    #data_dir = '~/cifar10_analysis/cifar10_analysis/cifar10_data/cifar10'
    #result_dir = '~/cifar10_analysis/cifar10_analysis/results/'

    data_dir = './cifar10_data/cifar10'
    result_dir = './results/'

    result_dir = os.path.expanduser(result_dir)

    # Get the current date and time
    current_time = datetime.datetime.now()

    # Format the date and time into a string
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    exp_res_dir_name = result_dir + hparams.experiment_name + '_' + str(time_string) +'/'
    os.mkdir(exp_res_dir_name)

    ###################################################################
    # Datasets Load
    ###################################################################
    print('-------------------------------------------------------------------------------')
    print('Datasets Load => start')
    print(' ')
    
    # Define type of preprocessing according to user input
    train_ds, test_ds = chose_preprocess(hparams.normalization,data_dir)

    # cifar-10 classes
    #cifar10_labels = ["aircraft", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    cifar10_labels = train_ds.classes

    ###################################################################
    # Train/Validation random split
    ###################################################################
    print(' ')
    print('-------------------------------------------------------------------------------')
    print('Train/Validation random split => start')
    print(' ')
    
    val_size = int(hparams.val_size * len(train_ds) / 100)
    train_size = len(train_ds) - val_size
    
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    ###################################################################
    # DataLoader
    ###################################################################
    print('-------------------------------------------------------------------------------')
    print('DataLoader => start')
    print(' ')

    batch_size = hparams.batch_size
    n_workers = hparams.num_workers

    train_dl, val_dl = dataloader_help(train_ds, val_ds, n_workers, batch_size, g, seed_worker)

    ###################################################################
    # To_device
    ###################################################################
    print('-------------------------------------------------------------------------------')
    print('To_device => start')
    print(' ')

    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    
    ###################################################################
    # Hyperparameters
    ###################################################################
    # Chose the Optimizer
    opt_func = chose_optimizer(hparams.optimization)

    # Chose the Model
    model = chose_model(hparams.model,device)
     
    ###################################################################
    # Train & save model
    ###################################################################
    print('-------------------------------------------------------------------------------')
    print('Train => start')
    print(' ')
    history, best_model_path, last_model_path = fit(hparams.epochs, hparams.lr, model, train_dl, val_dl, opt_func, exp_res_dir_name, hparams.experiment_name)
    print(' ')
    
    ###################################################################
    # Visualize trining => save images
    ###################################################################
    print('-------------------------------------------------------------------------------')
    print('Visualize trining => save images')
    print(' ')
    plot_losses(history,hparams.experiment_name,exp_res_dir_name)
    plot_accuracies(history,hparams.experiment_name,exp_res_dir_name)

    ###################################################################
    # Load the model
    ###################################################################
    print('-------------------------------------------------------------------------------')
    print('Load the model => start')
    print(' ')

    model_best = chose_model(hparams.model,device)
    model_best.load_state_dict(torch.load(best_model_path))

    model_last = chose_model(hparams.model,device)
    model_last.load_state_dict(torch.load(last_model_path))

    ###################################################################
    # Compare results of loaded model and working model => start
    ###################################################################
    print('-------------------------------------------------------------------------------')
    print('Check best/last models => start')
    print(' ')

    test_dl = DeviceDataLoader(DataLoader(test_ds, batch_size*2), device)

    eval_best = evaluate(model_best, test_dl)

    print("Summary result of test set => best model => val_loss: {:.4f}, val_acc: {:.4f}, val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f}".format(
            eval_best['val_loss'], eval_best['val_acc'], eval_best['val_precision'], eval_best['val_recall'], eval_best['val_f1']))

    eval_last = evaluate(model_last, test_dl)
    print("Summary result of test set => last model => val_loss: {:.4f}, val_acc: {:.4f}, val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f}".format(
            eval_last['val_loss'], eval_last['val_acc'], eval_last['val_precision'], eval_last['val_recall'], eval_last['val_f1']))
    print(' ')

    ###################################################################
    # Test set evaluation => save results for postprocessing
    ###################################################################
    print('-------------------------------------------------------------------------------')
    print('Test set evaluation (best model) => save results for postprocessing')
    print(' ')

    test_dl = DeviceDataLoader(DataLoader(test_ds, batch_size*2), device)
    
    file_name = hparams.experiment_name + "_test_set.json"
    experiment_name = hparams.experiment_name

    evaluate_summary(model_best, test_dl, experiment_name, file_name, device, exp_res_dir_name,cifar10_labels)

    print('-------------------------------------------------------------------------------')
    print('Valid set evaluation (best model) => save results for postprocessing')
    print(' ')
    file_name = hparams.experiment_name + "_val_set.json"
    experiment_name = hparams.experiment_name
    evaluate_summary(model_best, val_dl, experiment_name, file_name, device, exp_res_dir_name,cifar10_labels)

###################################################################
if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument("--normalization", default="No", type=str,help="No, N1, N2, N1_aug, N2_aug")
            # No - no normalization and augmentation
            # N1 - normalization only using mean/255 and std/255 of dataset 
            # N2 - normalization only using mean=[0, 0, 0] and std=[1, 1, 1] 
            # N1_aug - normalization N1 and augmentation => RandomHorizontalFlip & RandomCrop
            # N2_aug - normalization N2 and augmentation => RandomHorizontalFlip & RandomCrop
            # aug - augmentation => RandomHorizontalFlip & RandomCrop
    parser.add_argument("--val_size", default=10, type=int,help="% of validation set frtom training set")
    parser.add_argument("--batch_size", default=128, type=int,help="size of batch")
    parser.add_argument("--num_workers", default=multiprocessing.cpu_count(), type=int, help="Number of workers for dataloader")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train for")
    parser.add_argument("--optimization", default="Adam", type=str,help="Adam, SGD")
    parser.add_argument("--experiment_name", default="baseline", type=str,help="name for saving experimental results")
    parser.add_argument("--model", default="CNN", type=str,help="name of classification model")
            # CNN
            # ResNet_18
            # ResNet_34
            # ViT 
            # ViT_small
            # ViT_tiny
            # ViT_simple
    
    args = parser.parse_args()

    main(args)

    print('-------------------------------------------------------------------------------')
    print('END OF CODE')
    print('-------------------------------------------------------------------------------')
    print(' ')
    
    