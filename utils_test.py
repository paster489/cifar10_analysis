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





