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
from sklearn.metrics import precision_score, recall_score, f1_score

###################################################################
# Image Classification Base
###################################################################
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        #return {'val_loss': loss.detach(), 'val_acc': acc}

        # Calculate precision, recall, and F1-score
        prec, recall, f1 = precision_recall_f1(out, labels)  # Calculate precision, recall, F1
        return {'val_loss': loss.detach(), 'val_acc': acc, 'val_precision': prec, 'val_recall': recall, 'val_f1': f1}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        #return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

        # Combine precision, recall, and F1-score
        batch_precisions = [x['val_precision'] for x in outputs]
        epoch_precision = torch.tensor(batch_precisions).mean()
        
        batch_recalls = [x['val_recall'] for x in outputs]
        epoch_recall = torch.tensor(batch_recalls).mean()
        
        batch_f1s = [x['val_f1'] for x in outputs]
        epoch_f1 = torch.tensor(batch_f1s).mean()

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 
                'val_precision': epoch_precision.item(), 'val_recall': epoch_recall.item(), 'val_f1': epoch_f1.item()}
        
    
    
    def epoch_end(self, epoch, result):
        #print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        #    epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc']))
        
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f}".format(
            epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc'], result['val_precision'], result['val_recall'], result['val_f1']))

###################################################################
def precision_recall_f1(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    true_positives = torch.sum(preds == labels).item()
    false_positives = torch.sum(preds != labels).item()
    false_negatives = torch.sum(preds != labels).item()
    
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return precision, recall, f1

###################################################################
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
