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
    print("** accuracy: {:.4f}".format(accuracy_score(true_values, pred_values)))
    print("--")
    print("confusion matrix")
    print(confusion_matrix(true_values, pred_values))
    print("--")
    print("classification report")
    print(classification_report(true_values, pred_values, target_names=cifar10_labels))

    # save results
    metric_filepath = os.path.join(model_dir, file_name)
    write_metrics(true_values, pred_values, experiment_name, metric_filepath)

###################################################################
def evaluate_inference(model, test_dl, device):
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
from utils_hparams import *

def predict_image_inference(img, model, device, test_ds):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return test_ds.classes[preds[0].item()]

###################################################################
# Show 10 correct predictions
###################################################################
def show_correct_pred(test_ds_clean,test_ds,model_best_optimal,device):
  # Counter to keep track of displayed mismatched images
    displayed_count = 0

    # Calculate number of rows and columns for the subplot grid
    num_images = 10  # Number of mismatched images to display
    num_cols = 5     # Number of columns in the grid
    num_rows = math.ceil(num_images / num_cols)

    # Create the subplot grid
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    # Iterate through the test dataset
    for index in range(len(test_ds)):
        img, label = test_ds[index]
        img_clean, label_clean = test_ds_clean[index]
        
        predicted_label = predict_image_inference(img, model_best_optimal, device, test_ds)
        
        # Check if prediction does not match the actual label
        if predicted_label == test_ds.classes[label]:
            # Plot the image in the next available subplot
            ax = axs[displayed_count // num_cols, displayed_count % num_cols]
            ax.imshow(img_clean.permute(1, 2, 0))
            ax.set_title(f"Actual: {test_ds.classes[label]}\nPredicted: {predicted_label}")
            ax.axis('off')
            
            # Increment the counter
            displayed_count += 1
            
            # Break the loop if all desired images have been displayed
            if displayed_count == num_images:
                break

    # Hide any remaining empty subplots
    for i in range(displayed_count, num_rows * num_cols):
        axs[i // num_cols, i % num_cols].axis('off')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

###################################################################
# Show 10 incorrect predictions
###################################################################
def show_incorrect_pred(test_ds_clean, test_ds,model_best_optimal,device):
    # Counter to keep track of displayed mismatched images
    displayed_count = 0

    # Calculate number of rows and columns for the subplot grid
    num_images = 10  # Number of mismatched images to display
    num_cols = 5     # Number of columns in the grid
    num_rows = math.ceil(num_images / num_cols)

    # Create the subplot grid
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    # Iterate through the test dataset
    for index in range(len(test_ds)): 
        img, label = test_ds[index]
        img_clean, label_clean = test_ds_clean[index]
        
        predicted_label = predict_image_inference(img, model_best_optimal, device, test_ds)
        
        # Check if prediction does not match the actual label
        if predicted_label != test_ds.classes[label]:
            # Plot the image in the next available subplot
            ax = axs[displayed_count // num_cols, displayed_count % num_cols]
            
            ax.imshow(img_clean.permute(1, 2, 0))
            
            ax.set_title(f"Actual: {test_ds.classes[label]}\nPredicted: {predicted_label}")
            ax.axis('off')
            
            # Increment the counter
            displayed_count += 1
            
            # Break the loop if all desired images have been displayed
            if displayed_count == num_images:
                break

    # Hide any remaining empty subplots
    for i in range(displayed_count, num_rows * num_cols):
        axs[i // num_cols, i % num_cols].axis('off')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()