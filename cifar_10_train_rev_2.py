###################################################################
# Import libraries
###################################################################
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

from utils import *

###################################################################
# cifar-10 classes
###################################################################
cifar10_labels = ["aircraft", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

###################################################################
# Paths
###################################################################
data_dir = '~/cifar10_analysis/cifar10_data/cifar10'
model_dir = '~/cifar10_analysis/results/'
model_dir = os.path.expanduser(model_dir)

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
# Fit
###################################################################
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
##################################################################################

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    
    for epoch in range(epochs):
        
        # Training Phase 
        model.train()
        train_losses = []
        train_accs = []  # List to store train accuracies
        
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)

            acc = accuracy(model(batch[0]), batch[1])
            train_accs.append(acc)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_acc'] = torch.stack(train_accs).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        
    return history

###################################################################
# MAIN
###################################################################

def main(hparams):
    print('main start')
    print(' ')
    ###################################################################
    # Datasets Load
    ###################################################################
    print('Datasets Load')
    print(' ')
    
    if hparams.normalization == "No":
        train_ds, test_ds = simple_DL(data_dir)
    elif hparams.normalization == "N1":
        train_ds, test_ds = norm_1_DL(data_dir)
    elif hparams.normalization == "N2":
        train_ds, test_ds = norm_2_DL(data_dir)
    elif hparams.normalization == "N1_aug":
        train_ds, test_ds = norm_1_aug_DL(data_dir)
    elif hparams.normalization == "N2_aug":
        train_ds, test_ds = norm_2_aug_DL(data_dir)
    else:
        print('Error in normalization parameter input')

    ###################################################################
    # Train/Validation random split
    ###################################################################
    print('Train/Validation random split')
    print(' ')
    
    val_size_samples = int(hparams.val_size * len(train_ds) / 100)
    val_size = val_size_samples
    train_size = len(train_ds) - val_size_samples
    
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    ###################################################################
    # DataLoader
    ###################################################################
    print('DataLoader')
    print(' ')

    batch_size = hparams.batch_size
    n_workers = hparams.num_workers

    train_dl, val_dl = dataloader_help(train_ds, val_ds, n_workers, batch_size, g, seed_worker)

    ###################################################################
    # To_device
    ###################################################################
    print('To_device')
    print(' ')

    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    
    ###################################################################
    # Hyperparameters
    ###################################################################
    # Optimizer
    if hparams.optimization == 'Adam':
        opt_func = torch.optim.Adam
    elif hparams.optimization == 'SGD':
        opt_func = torch.optim.SGD
    else:
        print('Error in optimasation parameter input')

    # Model
    if hparams.model == 'CNN':
        from models.CNN_model import Cifar10CnnModel
        model = to_device(Cifar10CnnModel(), device)
    elif hparams.model == 'ResNet_18':
        from models.ResNet_model import ResNet, BasicBlock
        model = to_device(ResNet(BasicBlock, [2, 2, 2, 2]), device)
    elif hparams.model == 'ResNet_34':
        from models.ResNet_model import ResNet, BasicBlock
        model = to_device(ResNet(BasicBlock, [3, 4, 6, 3]), device)
    elif hparams.model == 'ViT':
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
     
    ###################################################################
    # Train
    ###################################################################
    print('Train')
    print(' ')
    history = fit(hparams.epochs, hparams.lr, model, train_dl, val_dl, opt_func)
    print(' ')
    
    ###################################################################
    # Visualize trining => save images
    ###################################################################
    print('Visualize trining => save images')
    print(' ')
    plot_losses(history,hparams.experiment_name,model_dir)
    plot_accuracies(history,hparams.experiment_name,model_dir)

    ###################################################################
    # Save the model
    ###################################################################
    print('Save the model')
    print(' ')
    torch.save(model.state_dict(), model_dir  + 'model_'+ hparams.experiment_name + '.pth')

    ###################################################################
    # Test set evaluation => save results for postprocessing
    ###################################################################
    print('Test set evaluation => save results for postprocessing')
    print(' ')

    test_dl = DeviceDataLoader(DataLoader(test_ds, batch_size*2), device)
    
    file_name = hparams.experiment_name + ".json"
    experiment_name = hparams.experiment_name
    evaluate_summary(model, test_dl, experiment_name, file_name, device, model_dir,cifar10_labels)


###################################################################
if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument("--normalization", default="No", type=str,help="No, N1, N2, N1_aug, N2_aug")
            # No - no normalization and augmentation
            # N1 - normalization only using mean/255 and std/255 of dataset 
            # N2 - normalization only using mean=[0, 0, 0] and std=[1, 1, 1] 
            # N1_aug - normalization N1 and augmentation => RandomHorizontalFlip & RandomCrop
            # N2_aug - normalization N2 and augmentation => RandomHorizontalFlip & RandomCrop
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
            # ViT - run
    
    args = parser.parse_args()

    main(args)

    print('END OF CODE')
    print(' ')
    
    # to run the code
    # python cifar_10_train_rev_2.py --normalization "No" --val_size 10 --batch_size 128 --num_workers 4 --lr 0.001 --epochs 10 --optimization "Adam" --experiment_name "baseline" --model "ResNet_18"


# Running description
# "ViT_No" => python cifar_10_train_rev_2.py --normalization "No" --val_size 10 --batch_size 128 --num_workers 4 --lr 0.001 --epochs 10 --optimization "Adam" --experiment_name "ViT_No" --model "ViT"


# AR => https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/