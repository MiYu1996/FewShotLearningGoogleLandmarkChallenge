import os
import pandas as pd
import numpy as np
import sklearn
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


class Landmarks_Dataset(Dataset):
    def __init__(self, X_path='dataset/train_images',
                 y_info='dataset/train_sample.csv',
                 mode = 'train',
                 paired = False):
        """
        X_path: path for training images
        y_info: list of two lists in shape [[image ids], [labels]] or path for train.csv
                if paired, then y_info should be [[(image_id1, image_id2)], [labels]]
        """
        assert mode in ['train', 'test']
        self.mode = mode
        self.paired = paired
        self.base_path = X_path
        if self.mode == 'train':
            self.transform = T.Compose([T.Resize(256),
                         T.RandomHorizontalFlip(),
                         T.RandomCrop(224),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        if self.mode == 'test':
            self.transform = T.Compose([T.Resize(256),
                         T.CenterCrop(224),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        if type(y_info) == str:
            df = pd.read_csv(y_info)
            self.id_list = df['id'].tolist()
            self.labels = np.array(df['landmark_id'].tolist(), dtype=int)
            print("Full training data initialized!")
        else:
            assert type(y_info) == list
            self.id_list = y_info[0]
            self.labels = y_info[1]
            print("Partial training data initialized!")
        if self.paired:
            assert len(self.id_list[0]) == 2

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        images = []
        label = self.labels[idx]
        for img_id in self.id_list[idx]:
            img_id = self.id_list[idx]
            img_path = os.path.join(self.base_path, img_id[0], img_id[1], img_id[2], img_id + '.jpg')
            image = Image.open(img_path)

            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        if self.paired:
            return images[0], images[1], label
        else:
            return images[0], label


class Landmarks_Label_based(Dataset):
    def __init__(self, X_path='dataset/train_images',
                 img_dict='dataset/train_sample.csv',
                 status='train',
                 mode='Prototypical',
                 N_S=3, N_Q=2, p=0.75):
        """
        X_path: path for training images
        img_dict: a dictionary that maps a image label to a list of image ids that belong to the given label
        mode: 'Prototypical' or 'Sia'
        N_S, N_Q: parameter for training prototypical networks
        p: parameter for training Siamese networks (probability of output being the same calss)
        """
        assert status in ['train', 'test', 'validation']
        assert mode in ['Prototypical', 'Siamese', 'MetaSVM']

        self.mode = mode
        self.status = status
        self.N_S = N_S
        self.N_Q = N_Q
        self.p = p
        self.base_path = X_path

        if self.status == 'train':
            self.transform = T.Compose([
                         T.Resize(256),
                         T.RandomHorizontalFlip(),
                         T.RandomCrop(224),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                         ])
        else:
            self.transform = T.Compose([
                         T.Resize(256),
                         T.CenterCrop(224),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                         ])
        if type(img_dict) == str:
            df = pd.read_csv(img_dict)
            self.img_dict = df.groupby('landmark_id')['id'].apply(list).to_dict()
            self.labels = list(self.img_dict.keys())
        else:
            assert type(img_dict) == dict
            self.img_dict = img_dict
            self.labels = list(img_dict.keys())
        print("Data initialized!")

    def __len__(self):
        return len(self.labels)

    def id_to_img(self, img_id):
        img_path = os.path.join(self.base_path, img_id[0], img_id[1], img_id[2], img_id + '.jpg')
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __getitem__(self, idx):
        target_class = self.labels[idx]
        id_lists = self.img_dict[target_class]
        if self.status in ['train', 'test']:
            random.shuffle(id_lists)
        if self.mode in ['Prototypical', 'MetaSVM']:
            S_k = torch.stack([self.id_to_img(img_id) for img_id in id_lists[0:self.N_S]])
            if self.N_S < len(id_lists):
                Q_k = torch.stack([self.id_to_img(img_id) for img_id in id_lists[self.N_S:self.N_S + self.N_Q]])
            else:
                Q_k = torch.ones(2, 2, 2, 2, 2)
            return S_k, Q_k, target_class
        else:
            if np.random.rand() <= self.p:
                image_1 = self.id_to_img(id_lists[0])
                image_2 = self.id_to_img(id_lists[1])
                return image_1, image_2, True
            else:
                image_1 = self.id_to_img(id_lists[0])
                other_class = random.choice(self.labels)
                image_2 = self.id_to_img(random.choice(self.img_dict[other_class]))
                return image_1, image_2, target_class == other_class


class Landmarks_Test(Dataset):
    def __init__(self, X_path='dataset/train_images',
                 y_info='dataset/new_test.csv',
                 transform=T.ToTensor()):
        """
        X_path: path for training images
        y_info: list of two lists in shape [[image ids], [labels]] or path for test.csv
        """
        self.base_path = X_path
        self.transform = transform
        if type(y_info) == str:
            df = pd.read_csv(y_info)
            self.id_list = df['id'].tolist()
            self.labels = np.array(df['landmark_id'].tolist(), dtype=int)
            print("Full testing data initialized!")
        else:
            assert type(y_info) == list
            self.id_list = y_info[0]
            self.labels = y_info[1]
            print("Partial traintestinging data initialized!")

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        img_id = self.id_list[idx]
        label = self.labels[idx]
        img_path = os.path.join(self.base_path, img_id[0], img_id[1], img_id[2], img_id + '.jpg')
        image = Image.open(img_path)
        image = T.functional.resize(image, (224, 224))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

# class Landmarks_Test_I(Dataset):
#     def __init__(self, X_path='dataset/test_images', transform=T.ToTensor()):
#         self.base_path = X_path
#         files_list = os.listdir(X_path)
#         self.id_list = [file_name[0:-4] for file_name in files_list]
#         self.transform = transform

#     def __len__(self):
#         return len(self.id_list)

#     def __getitem__(self, idx):
#         img_id = self.id_list[idx]
#         img_path = os.path.join(self.base_path, img_id + '.jpg')
#         image = Image.open(img_path)

#         if self.transform is not None:
#             image = self.transform(image)

#         return image, img_id


# def get_dataset(val_size=0.01,
#                 X_path='dataset/train_images/',
#                 y_path='dataset/new_train.csv',
#                 transform=F.ToTensor(),
#                 shuffle=False):
#     """
#     val_size: size for validation data set
#     X_path: path for training images
#     y_path: path for train.csv
#     shuff: whether shuffle the data
#     """
#     df = pd.read_csv(y_path)
#     if shuffle:
#         df = sklearn.utils.shuffle(df)
#     y_info = [df['id'].tolist(), np.array(df['landmark_id'].tolist(), dtype=int)]
#     split_idx = int((1 - val_size) * len(y_info[0]))
#     train_info = [y_info[0][0:split_idx], y_info[1][0:split_idx]]
#     val_info = [y_info[0][split_idx + 1:], y_info[1][split_idx + 1:]]
#     train_set = Landmarks_Train(X_path, train_info, transform)
#     validation_set = Landmarks_Train(X_path, val_info, transform)
#     return train_set, validation_set


# new loader function that works for our baseline
def get_dataset(val_size=0.01,
                X_path='dataset/train_images/',
                y_train_path='dataset/new_train.csv',
                y_test_path = 'dataset/new_test.csv',
                shuffle=False):
        """
        val_size: size for validation data set
        X_path: path for training images
        y_path: path for train.csv
        shuff: whether shuffle the data
        """
        df_train = pd.read_csv(y_train_path)
        df_test = pd.read_csv(y_test_path)
        if shuffle:
            df_train = sklearn.utils.shuffle(df_train)
            df_test = sklearn.utils.shuffle(df_test)
        y_train_info = [df_train['id'].tolist(), np.array(df_train['landmark_id'].tolist(), dtype=int)]
        split_idx = int((1 - val_size) * len(y_train_info[0]))
        train_info = [y_train_info[0][0:split_idx], y_train_info[1][0:split_idx]]
        val_info = [y_train_info[0][split_idx + 1:], y_train_info[1][split_idx + 1:]]
        test_info = [df_test['id'].tolist(), np.array(df_test['landmark_id'].tolist(), dtype=int)]
        train_set = Landmarks_Train(X_path, train_info)
        validation_set = Landmarks_Train(X_path, val_info)
        test_set = Landmarks_Test(X_path, test_info)
        return train_set, validation_set, test_set


# def test_get_data():
#     train_set, val_set = get_dataset(shuffle=True)
#     train_loader = DataLoader(train_set, batch_size=12, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_set, batch_size=12, shuffle=True, num_workers=0)

#     iteration = 0
#     print(f"Training dataset size: {len(train_set)}.")
#     for batch_idx, (data, labels) in enumerate(train_loader):
#         print(labels)
#         print(data.shape)

#         iteration += 1
#         if iteration >= 5:
#             break

#     iteration = 0
#     print(f"Validation dataset size: {len(val_set)}.")
#     for batch_idx, (data, label) in enumerate(val_loader):
#         print(label)
#         print(data.shape)

#         iteration += 1
#         if iteration >= 5:
#             break


def Load_data():
    train_set, val_set, test_set = get_dataset(val_size = 0, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=12, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=12, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    print("Data Loaders built successfully!")

    return dataloaders
