import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_loader import Landmarks_Label_based
from utils import simple_train
from models import SiameseNetworks
from torchvision import models


# def pretrained(device=torch.device('cuda')):
#     model = models.resnet50(pretrained=True)
#     # for params in model.parameters():
#     #     params.requires_grad = False
#     model = model.to(device)
#     res50_conv = torch.nn.Sequential(*list(model.children())[:-2])
#     return res50_conv

base_CNN = nn.Sequential(
        nn.Conv2d(3, 64, 10),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 7),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 128, 4),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 128, 4),
        nn.BatchNorm2d(128),
        nn.ReLU()
)

df = pd.read_csv('dataset/train_sample.csv')
img_dict = df.groupby('landmark_id')['id'].apply(list).to_dict()
small_dict = {}
num = 0
for key in img_dict.keys():
    small_dict[key] = img_dict[key]
    num += 1
    if num >= 100:
        break
print("small dataset initialized!")

training_set = Landmarks_Label_based(img_dict=small_dict, mode='Siamese')
val_set = Landmarks_Label_based(img_dict=small_dict, mode='Siamese', status='test', p=0.5)
train_loader = DataLoader(training_set, batch_size=10)
val_loader = DataLoader(val_set, batch_size=10)
siamese_net = SiameseNetworks(base_CNN, 128 * 20 * 20, 2048)
trainable_para = [params for params in siamese_net.parameters() if params.requires_grad == True]

optimizer = torch.optim.Adam(trainable_para, lr=1e-4, weight_decay=0.1)
simple_train(siamese_net, train_loader, val_loader, optimizer, 'trial_train/naive_siamese.pth',
            loss_func=torch.nn.BCELoss(), mode='Siamese', epochs=150)
