import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data_loader import Landmarks_Label_based
from utils import simple_train
from models import *
from torchvision import models

# class Base_Net(nn.Module):
#     def __init__(self, CNN_part):
#         super().__init__()
#
#         self.CNN_part = CNN_part
#         self.fc = nn.Linear(2048*7*7, 1024)
#         self.norm = nn.BatchNorm1d(1024)
#
#     def forward(self, x):
#         out = self.CNN_part(x)
#         out = flatten(out)
#         out = self.fc(out)
#         out = self.norm(out)
#         out = F.relu(out)
#         return out
#
# def pretrained(device=torch.device('cuda')):
#     model = models.resnet50(pretrained=True)
#     for params in model.parameters():
#         params.requires_grad = False
#     model = model.to(device)
#     res50_conv = torch.nn.Sequential(*list(model.children())[:-2])
#     return res50_conv

# res50 = pretrained()
# base_net = Base_Net(res50)
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

# df = pd.read_csv('dataset/train_sample.csv')
# img_dict = df.groupby('landmark_id')['id'].apply(list).to_dict()
# small_dict = {}
# num = 0
# for key in img_dict.keys():
#     small_dict[key] = img_dict[key]
#     num += 1
#     if num >= 100:
#         break
# print("small dataset initialized!")

training_set = Landmarks_Label_based(img_dict='dataset/train_sample.csv', mode='Prototypical')
val_set = Landmarks_Label_based(img_dict='dataset/validation_sample.csv', mode='Prototypical', status='test', N_S=4, N_Q=1)
train_loader = DataLoader(training_set, batch_size=20)
val_loader = DataLoader(val_set, batch_size=10)
proto_net = PrototypicalNetworks(base_net, nn.PairwiseDistance())
trainable_para = [params for params in proto_net.parameters() if params.requires_grad == True]
optimizer = torch.optim.Adam(trainable_para, lr=5e-4)

simple_train(proto_net, train_loader, val_loader, optimizer, 'trial_train/naive_proto.pth',
            loss_func=nn.CrossEntropyLoss(), mode = 'Prototypical', epochs=3, print_interval=100)
