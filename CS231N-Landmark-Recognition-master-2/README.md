# CS231N-Landmark-Recognition

The "train.csv" can be downloaded at [`https://s3.amazonaws.com/google-landmark/metadata/train.csv`](https://s3.amazonaws.com/google-landmark/metadata/train.csv). In order for EDA.ipynb to work properly, it should be put at 'dataset/train.csv'.

data_loader.py: the modified dataloader that sample the batches of images for training

EDA_plot.py: the plots for exploratory data analysis

EDA.ipynb: exploratory data analysis before applying the neural network

landmark16, landmark20, landmark64, landmark100: mini-landmark challenge dataset with 16 classes, 20 classes, 64 classes, and 100 classes respectively

metaSVM_train.py: the file that is used for training of metaSVM method

models_proj.py: includes the baseline models

models.py: includes the implementation of metaSVM, siamese, and prototypical methods

prototypical_test.py: the file that runs the test for prototypical method

prototypical_train.py: the file that runs the training for prototypical method

siamese_train.py: the file that runs the training for siamese network

Super_Small_Size_Data.ipynb: this notebook is used to generate very small dataset to do test run for the methods

utils.py: includes evaluation functions and function that checks the accuracy
