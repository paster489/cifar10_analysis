# cifar10 classifyer training

1\. Reproduce conda environment using torch_gpu_env.yml file:

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

4\. To run the training use the python file "cifar_10_train_rev_2.py".   

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

--val_size    
&emsp;&emsp;Size of validation set, in % from the total training set.

--batch_size  
&emsp;&emsp;Size of batch. The validation batch sixe is x2 of the training batch size.

--num_workers  
&emsp;&emsp;Number of workers for dataloader. Don't use high number => can lead to bottle neck.

--lr  
&emsp;&emsp;Learning rate.

--epochs  
&emsp;&emsp;Number of training epochs.

--optimization  
&emsp;&emsp;"Adam"  
&emsp;&emsp;"SGD"  

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

5\. The analysis of the dataset is in the file “data_visualization.ipynb”.  

6\. The summary of model performance using the test set is in the file “inference.ipynb”.  

7\. Models are inside "models" folder.

8\. The summary, conclusions and theoretical questions are inside pdf file.

9\. The reults of experimental running are inside "results" and "LSF_out" folders. 

