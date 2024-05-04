# MCMVDRP
The source code of paper "A Multi-Channel Multi-View Deep Learning Framework for Cancer Drug Response Prediction"
# Resources:
* README.md: this file.
* data: GDSC dataset
## source codes:
* preprocess.py: create data in pytorch format.
* utils.py: include TestbedDataset used by create_data.py to create data, performance measures and functions to draw loss, pearson by epoch.
* models/gat_gcn.py: proposed models GAT_GCN receiving graphs as input for drugs.
* training.py: train a MCMMDRP model.

# Dependencies
* Torch
* Pytorch_geometric
* Rdkit
* Matplotlib
* Pandas
* Numpy
* Scipy
# Step-by-step running:
* Create data in pytorch format
```
python preprocess.py --choice 0 
```
choice:    0: create mixed test dataset     1: create saliency map dataset     2: create blind drug dataset      3: create blind cell dataset  
  
This returns file pytorch format (.pt) stored at data/processed including training, validation, test set.
* Train a GraphDRP model
```
python training.py --model 0 --train_batch 256 --val_batch 256 --test_batch 256 --lr 0.0005 --num_epoch 300 --log_interval 20 --cuda_name "cuda:0"
```
model:       0: GINConvNet       1: GAT_GCN  
  
To train a model using training data. The model is chosen if it gains the best MSE for testing data.  

This returns the model and result files for the modelling achieving the best MSE for testing data throughout the training.
