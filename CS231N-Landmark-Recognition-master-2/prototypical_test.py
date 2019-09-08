import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data_loader import *
from utils import *
from models import *

base_net = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        Flatten()
)

training_set = Landmarks_Label_based(img_dict='dataset/train_sample.csv', mode='Prototypical', N_S=5, N_Q=0)
test_set = Landmarks_Dataset(y_info='dataset/proto_valset.csv')
train_loader = DataLoader(training_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=1)
proto_net = PrototypicalNetworks(base_net, nn.PairwiseDistance())
state = torch.load('trial_train/naive_proto.pth')
proto_net.load_state_dict(state['state_dict'])

acc = test_accuracy_proto(proto_net.base_net, train_loader, test_loader, training_set.labels)
