# CIFAR10 CLASSIFIER TRAINING

# HARDWARE

The work was executued on servers:
    
&emsp;&emsp;GPU = NVIDIA A40, 8 x 46 GB  
&emsp;&emsp;Model = ProLiant XL675d Gen10 Plus   
&emsp;&emsp;Server Cores = 112  
&emsp;&emsp;CPU =Version: AMD EPYC 7453 28-Core Processor  
&emsp;&emsp;OS = Centos 7

# WORKING ENVIRONMENT

1\. Reproduce conda environment using "torch_gpu_env.yml" file:

```ruby
conda env create --name envname --file=torch_gpu_env.yml
```

2\. Use the next vesions of modules:    
     
&emsp;&emsp;  a) NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.8.0  
&emsp;&emsp;  b) CUDA/11.8.0

3\. Activate conda environment:
```ruby
conda activate envname
```
# GIT REPOSITORY CLONE

```ruby
git clone git@github.com:paster489/cifar10_analysis.git
```


# DATA VISUALIZATION
The analysis of the dataset is in the file “data_visualization.ipynb”.

# TRAINING

To run the training use the python file "cifar_10_train_rev_2.py".   

Define input arguments.
For example:

```ruby
python cifar_10_train_rev_2.py --normalization "No" --val_size 10 --batch_size 128 --num_workers 4 --lr 0.001 --epochs 100 --optimization "Adam" --experiment_name "CNN_No_run_1" --model "CNN"
```

 --normalization  
&emsp;&emsp;"No" - no normalization and augmentation  
&emsp;&emsp;"N1" - normalization only, using mean/255 and std/255 of dataset   
&emsp;&emsp;"N2" - normalization only, using mean=[0, 0, 0] and std=[1, 1, 1]   
&emsp;&emsp;"N1_aug" - normalization N1 and augmentation using RandomHorizontalFlip & RandomCrop   
&emsp;&emsp;"N2_aug" - normalization N2 and augmentation using RandomHorizontalFlip & RandomCrop  
&emsp;&emsp;"aug" - augmentation only, using RandomHorizontalFlip & RandomCrop  

- "N1_aug" => the optimal one.

--val_size    
&emsp;&emsp;Size of validation set, in % from the total training set.  

- 10 => the optimal one.

--batch_size  
&emsp;&emsp;Size of batch. The validation batch sixe is x2 of the training batch size.  

- 256 => the optimal one.

--num_workers  
&emsp;&emsp;Number of workers for dataloader. Don't use high number => can lead to bottle neck.

- 4 => the optimal one.

--lr  
&emsp;&emsp;Learning rate.

- 0.001 => the optimal one.

--epochs  
&emsp;&emsp;Number of training epochs.

- 30 => the optimal one.

--optimization  
&emsp;&emsp;"Adam"  
&emsp;&emsp;"SGD"  

- "Adam" => the optimal one.

--experiment_name  
&emsp;&emsp;Name of the experimental run. Under this name the directory in results. 
filder will be cretead where the results of trainijg will be saved.  

--model   
&emsp;&emsp;Name of the training model: 

&emsp;&emsp;&emsp;&emsp;"CNN"  
&emsp;&emsp;&emsp;&emsp;"ResNet_18"  
&emsp;&emsp;&emsp;&emsp;"ResNet_34"  
&emsp;&emsp;&emsp;&emsp;"ViT"   
&emsp;&emsp;&emsp;&emsp;"ViT_small"  
&emsp;&emsp;&emsp;&emsp;"ViT_tiny"  
&emsp;&emsp;&emsp;&emsp;"ViT_simple"  

- "ResNet_34" => the optimal one.

  
# INFERENCE
The summary of model performance, using the test set, is in the file “inference.ipynb”.  

# GENERAL
1\. Models are inside "models" folder.

2\. The summary, conclusions and theoretical questions are inside pdf file.

3\. The results of experimental running are inside "results" and "LSF_out" folders.  

4\. "LSF_err" folder shows the errors during the job run. 

5\. Work was executed on LSF job scheualer cluster. 

6\. In the file "run-gpu-torch-cifar10.lsf" there are definitions for LSF jobs.

