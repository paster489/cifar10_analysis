# cifar10_analysis

# to run the code

python cifar_10_train_rev_2.py --normalization "No" --val_size 10 --batch_size 128 --num_workers 4 --lr 0.001 --epochs 15 --optimization "Adam" --experiment_name "test" --model "CNN"


# Running description
# "ViT_No" => python cifar_10_train_rev_2.py --normalization "No" --val_size 10 --batch_size 128 --num_workers 4 --lr 0.001 --epochs 10 --optimization "Adam" --experiment_name "ViT_No" --model "ViT"


# AR => https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/