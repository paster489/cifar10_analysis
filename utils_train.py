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
import matplotlib
matplotlib.use('Agg')  # Use Agg backend

###################################################################
# Accuracy
###################################################################
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

###################################################################
# Evaluate
###################################################################
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

###################################################################
# Fit
###################################################################
def fit(epochs, lr, model, train_loader, val_loader, opt_func, exp_res_dir_name, experiment_name):
    
    best_model_path = exp_res_dir_name  + 'model_'+ experiment_name + '_best.pth'
    last_model_path = exp_res_dir_name  + 'model_'+ experiment_name + '_last.pth'

    history = []
    optimizer = opt_func(model.parameters(), lr)
    
    # Variables to track best model
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        train_accs = []

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

        # Save model if validation loss is improved
        if result['val_loss'] < best_val_loss:
            best_val_loss = result['val_loss']
            torch.save(model.state_dict(), best_model_path)

    # Save the last epoch model
    torch.save(model.state_dict(), last_model_path)

    return history, best_model_path, last_model_path

###################################################################
# Plot  training loss/acc
###################################################################
def plot_accuracies(history,experiment_name,model_dir):
    val_acc = [x['val_acc'] for x in history]
    train_acc = [x['train_acc'] for x in history]
    plt.figure()
    plt.plot(train_acc, '-bx')
    plt.plot(val_acc, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Training', 'Validation'])
    plt.title('Accuracy vs. No. of epochs');
    plt.savefig(model_dir + experiment_name + '_acc.png')
    plt.close()

def plot_losses(history,experiment_name,model_dir):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.figure()
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
    plt.savefig(model_dir  + experiment_name + '_loss.png')
    plt.close()




