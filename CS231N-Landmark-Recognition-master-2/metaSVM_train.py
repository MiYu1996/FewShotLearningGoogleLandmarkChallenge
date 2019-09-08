import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data_loader import Landmarks_Label_based
from utils import simple_train
from models import *


training_set = Landmarks_Label_based(img_dict='dataset/train_sample.csv', mode='MetaSVM')
val_set = Landmarks_Label_based(img_dict='dataset/validation_sample.csv', mode='MetaSVM', status='test', N_S=4, N_Q=1)
train_loader = DataLoader(training_set, batch_size=10)
val_loader = DataLoader(val_set, batch_size=10)

base_net = ResNet12()
base_net.to(device=torch.device('cuda'))
meta_svm = MetaSVMNetworks(base_net)
optimizer = torch.optim.Adam(meta_svm.parameters(), lr=1e-4)

simple_train(meta_svm, train_loader, val_loader, optimizer, 'trial_train/naive_metasvm.pth', 'tria_train/best_naive_metasvm.pth',
            loss_func=nn.CrossEntropyLoss(), mode='MetaSVM', epochs=3, print_interval=100)
