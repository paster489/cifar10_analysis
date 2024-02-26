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

    train_ds_norm_1, test_ds_norm_1 = norm_1_DL()

    
    return train_ds_norm_1_aug, test_ds_norm_1

###################################################################
# Normalization 2 + Augmanation
###################################################################
def norm_2_aug_DL(data_dir):
    train_ds_norm_2, test_ds_norm_2 = norm_2_DL()
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
# DataLoader
###################################################################
def dataloader_help(train_ds, val_ds,n_workers,batch_size,g,seed_worker):
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=n_workers, pin_memory=True, generator=g, worker_init_fn=seed_worker)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=n_workers, pin_memory=True, generator=g, worker_init_fn=seed_worker)
        
    return train_dl, val_dl

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

###################################################################
# Accuracy
###################################################################
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


###################################################################
# Test set evaluation
###################################################################
def evaluate_F1(model, test_dl, device):
  model.eval()
  true_values, pred_values = [], []
    
  for batch in test_dl:
    images, labels = batch
    images = images.to(device)

    with torch.no_grad():
      logits = model(images)
        
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    true_values.extend(labels.tolist())
    pred_values.extend(preds.tolist())

  return true_values, pred_values
###################################################################
def write_metrics(true_values, pred_values, experiment_name, metric_filepath):
  fout = open(metric_filepath, "w")
  p, r, f, s = precision_recall_fscore_support(true_values, pred_values, average="micro")
  metrics_dict = {
    "precision": p, "recall": r, "f1-score": f, "support": s
  }
  metrics_dict["name"] = experiment_name
  fout.write(json.dumps(metrics_dict))
  fout.close()
###################################################################
def evaluate_summary(model, test_dl, experiment_name, file_name, device,model_dir,cifar10_labels):
    true_values, pred_values = evaluate_F1(model, test_dl,device)

    # print results
    print("** accuracy: {:.3f}".format(accuracy_score(true_values, pred_values)))
    print("--")
    print("confusion matrix")
    print(confusion_matrix(true_values, pred_values))
    print("--")
    print("classification report")
    print(classification_report(true_values, pred_values, target_names=cifar10_labels))

    # save results
    metric_filepath = os.path.join(model_dir, file_name)
    write_metrics(true_values, pred_values, experiment_name, metric_filepath)





