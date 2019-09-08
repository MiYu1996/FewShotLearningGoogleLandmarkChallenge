import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
#This is used to get the pre-trained weights
from torchvision import models
import data_loader
import utils

import numpy as np



###Distance metric
def Lp(x,y,p):
    return (torch.sum((x-y)**p))**(1.0/p)
    




### Use pretrained resnet-50 model
def pretrained(device):
    model = models.resnet50(pretrained=True)
    #Freeze model weights
    for params in model.parameters():
        params.requires_grad = False
    model = model.to(device)
    res50_conv = nn.Sequential(*list(model.children())[:-2])
    return res50_conv



def model_output(model, data):
    #evaluation mode
    model.eval()
    output = model(data)
    output = torch.mean(output, dim = 1)
    #check the dimension
    #print(output.shape)
    #flatten the dimension to calculate the distance
    output = torch.flatten(output, start_dim = 1)
    #print(output.shape)
    return output
        
        
###
### The baseline function:
### No training mode for this function 
### The dataloader needs to be a dictionary, where
### dataloader['val'] contains the validation images
### and dataloader['test'] contains the test images
### if p is set to be the distance metrics for different power
### cos_sim = 1, then use cosine similarity


### Input: Dataloaders: A dictionary ('train','val','test'):(image,label),  
### 
###  model pretrained model
###
### p = 2, the power of the distance metric
###
def baseline(dataloaders, model, p = 2):
    #want to check what the feature looks like after
    #passing through the model
    #Keep a running sum for the total accuracy
    total = 0.0
    total_size = 0.0
    #This is set to test this number of batches for accuracy
    #This function gives the validation error
    #when there is no label, can modify this function
    #to get the test class
    test_num_batch = 0
    label = [] 
    scores = []
    true_label = []
    for data_valid, targets_valid in dataloaders['val']:
        valid_batch_size = data_valid.shape[0]
        #get the valid_data
        print('Test batch ' + str(test_num_batch))
        data_valid = data_valid.to('cuda')
        out_valid = model_output(model, data_valid)
        min_distance = torch.ones(out_valid.shape[0], dtype=torch.float)
        result = torch.ones(out_valid.shape[0], dtype=torch.long)
        #Use a very large number first
        min_distance = min_distance.fill_(1e10)
        min_distance = min_distance.to('cuda')
        #Now compare each using 1-NN, and record the value
        batch_num = 0
        for data,targets in dataloaders['train']:
            print('Training batch ' + str(batch_num))
            data = data.to('cuda')
            out = model_output(model, data)
            #loop through the two datasets
            #for i in range(data_valid.shape[0]):
            for j in range(data.shape[0]):
                    #distance = Lp(out[j],out_valid[i],p)
                    #distance = Lp(data[j],data_valid[i],p)
                    #if distance == 0:
                        #print(Lp(out[j],out_valid[i],p))
                        #print(out[j])
                        #print(out_valid[i])
                    #if distance < min_distance[i]:
                        #min_distance[i] = distance
                        #result[i] = targets[j]
            #Using different distance metrics
            #repeat the same columns several times for calculation
                repeated_out = out[j].repeat(data_valid.shape[0],1)
                #print(out[j])
                #print(repeated_out)
                #print(repeated_out.shape)
                distance = F.pairwise_distance(out_valid,repeated_out,p)
                #print(distance)
                index = distance < min_distance
                min_distance[index] = distance[index]
                result[index] = targets[j]
            print('current test batch accuracy ' + 
                  str(1.0 - (torch.sum(result != targets_valid).double()/valid_batch_size).item()))
            #print(min_distance)
            batch_num += 1
        test_num_batch += 1
        total_size += valid_batch_size
        total += torch.sum(result != targets_valid).double()
        print('Total test batch accuracy ' + 
              str(1.0 - (total/total_size).item()))
        label = label + result.data.cpu().numpy().tolist()
        scores = scores + (1.0/ (min_distance.data.cpu().numpy() + 10e-6)).tolist()
        #print(min_distance.data.cpu().numpy() + 10e-6) 
        #sanity check for scores
        #print(scores)
        true_label = true_label + targets_valid.data.cpu().numpy().tolist()
        #get the GAP score
        print("Total GAP Score is" + 
              str(utils.GAP_evaluation(np.asarray(true_label),np.asarray(label),np.asarray(scores))))
        output = list(zip(true_label,label,scores))
        #print(label)
        #if test_num_batch > 1:
            #break
    #write the solutions into a .csv file
    out = open('output.csv', 'w')
    for row in output:
        for index in range(len(output[0])):
            out.write('%d;' % row[index])
        out.write('\n')
    out.close()
    #np.savetxt('output.csv', label,scores)
        
        

