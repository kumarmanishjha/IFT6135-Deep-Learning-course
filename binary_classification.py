# -*- coding: utf-8 -*-
"""Cat_Dog_classification

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1naQskegSzCq6joiqLmZruW5ERruUeh_-
"""

#Data Creation:

from PIL import Image

def rotate(image_path, degrees_to_rotate, saved_location):
    """
    Rotate the given photo the amount of given degreesk, show it and save it
    @param image_path: The path to the image to edit
    @param degrees_to_rotate: The number of degrees to rotate the image
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.rotate(degrees_to_rotate)
    rotated_image.save(saved_location)
    #rotated_image.show()

def flip_image(image_path, saved_location):
    """
    Flip or mirror the image
    @param image_path: The path to the image to edit
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_image.save(saved_location)
    #rotated_image.show()

import PIL

im = PIL.Image.open('/u/bhardwas/Documents/angry-cat-outside.jpg')
im.point(lambda x: x * 2)
im.save('/u/bhardwas/Documents/output_filename.jpg')

#Flip Image and create 10000 more:

cat = '/u/bhardwas/Documents/DL/Kaggle_cat_dog/ift6135h19/trainset/Cat/'
dog = '/u/bhardwas/Documents/DL/Kaggle_cat_dog/ift6135h19/trainset/Dog/'

for i in range(1,10000):
  flip_image(cat+str(i)+str('.Cat.jpg'), cat+str(i+10000)+str('.Cat.jpg'))
  
  
for i in range(1,10000):
  flip_image(dog+str(i)+str('.Dog.jpg'), dog+str(i+10000)+str('.Dog.jpg'))

for deg in degree:
  rotate('/u/bhardwas/Documents/angry-cat-outside.jpg', deg, '/u/bhardwas/Documents/'+str(deg)+'_test.jpg')

degree = [5,10,15,20,25,30]

rev_deg = [-5,-10,-15,-20,-25,-30]

#Rotate Cat:
count = 0
for deg in rev_deg:
  count = count + 1
  for j in range(0+count,10000,6):
    rotate(cat+str(j)+str('.Cat.jpg'), deg, cat+str(j+30000)+str('.Cat.jpg'))

#Rotate Dog:
count = 0
for deg in rev_deg:
  count = count + 1
  for j in range(0+count,10000,6):
    rotate(dog+str(j)+str('.Dog.jpg'), deg, dog+str(j+30000)+str('.Dog.jpg'))

#######################################################################################################################################
######################################################### Current Best Model ##########################################################
#######################################################################################################################################
import os
import warnings
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import random
import itertools
import imageio
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F

#Data Directories:
data_dir = '/u/bhardwas/Documents/DL/Kaggle_cat_dog/ift6135h19/trainset/'
val_dir = '/u/bhardwas/Documents/DL/Kaggle_cat_dog/cat_dog_data/'

#Parameters:
batch_size = 1
num_classes = 2

#Data Augmentation:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

trainloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(data_dir,
                         transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True)


testloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(val_dir,
                             transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size= batch_size,
        shuffle=True,
        num_workers=2,
pin_memory=True)


#Classification Model:
class Classifier(nn.Module):
    """Convnet Classifier"""
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.clf1 = nn.Linear(in_features = 256*14*14, out_features = 256)
        self.clf2 = nn.Linear(256, 64)
        self.clf3 = nn.Linear(64, out_features = num_classes)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        #print('Tensor shape:', x.shape)
        x = self.conv(x).squeeze()
        #print('After CONV2d shape:', x.shape)
        x = x.view(-1, 256*14*14)
        #print('After Reshape shape:', x.shape)
        out = F.relu(self.clf1(x))
        out = F.relu(self.clf2(out))
        out = F.relu(self.clf3(out))
        #out = self.sigmoid(out)
        #print('After FillyConn shape:', out.shape)
        return out


#Initializing Model:
clf = Classifier()
print(clf) 

#Cude setup:
cuda_available = torch.cuda.is_available()
cuda_available

if cuda_available:
    clf = clf.cuda()

#Setting up loss and optimizer funtions:
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(clf.parameters(), lr=0.001)

#Training and prediction step:
for epoch in range(50):
    losses = []
    train_total = 0
    train_correct = 0
    # Train
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = clf(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.data)
        #Training Accuracy:
        _, train_predicted = torch.max(outputs.data, 1)
        train_total += targets.size(0)
        train_correct += train_predicted.eq(targets.data).cpu().sum()

    print('Epoch : %d Loss : %.3f Training Acc : %.3f' % (epoch, torch.mean(torch.stack(losses)), 100.*train_correct/train_total))

    # Evaluate
    clf.eval()
    result = []
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
        outputs = clf(inputs)
        #Test Accuracy:
        _, predicted = torch.max(outputs.data, 1)
        result.append(predicted)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('Test Acc : %.3f' % (100.*correct/total))
    print('--------------------------------------------------------------')
    clf.train()

#Saving model and results:
'''
torch.save(clf.state_dict(), '/u/bhardwas/Documents/DL/Kaggle_cat_dog/model_3000')

clf = Classifier()
clf.load_state_dict(torch.load('/u/bhardwas/Documents/DL/Kaggle_cat_dog/model_3000'))
clf.cuda()
clf.eval()

import pandas as pd
final_out = pd.DataFrame(torch.stack(result).cpu().numpy())
final_out.to_csv('/u/bhardwas/Documents/DL/Kaggle_cat_dog/ift6135h19/results.csv')

'''


#######################################################################################################################################
######################################################### END #########################################################################
#######################################################################################################################################

#######################################################################################################################################
######################################################### Testing Model ##########################################################
#######################################################################################################################################
import os
import warnings
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import random
import itertools
import imageio
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F

#Data Directories:
data_dir = '/u/bhardwas/Documents/DL/Kaggle_cat_dog/ift6135h19/trainset/'
val_dir = '/u/bhardwas/Documents/DL/Kaggle_cat_dog/ift6135h19/testset/'

#Parameters:
batch_size = 1
num_classes = 2

#Data Augmentation:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

trainloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(data_dir,
                         transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True)


testloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(val_dir,
                             transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size= batch_size,
        shuffle=True,
        num_workers=2,
pin_memory=True)


#Classification Model:
class Classifier(nn.Module):
    """Convnet Classifier"""
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.clf1 = nn.Linear(in_features = 256*14*14, out_features = 256)
        self.clf2 = nn.Linear(256, 64)
        self.clf3 = nn.Linear(64, out_features = num_classes)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        #print('Tensor shape:', x.shape)
        x = self.conv(x).squeeze()
        #print('After CONV2d shape:', x.shape)
        x = x.view(-1, 256*14*14)
        #print('After Reshape shape:', x.shape)
        out = F.relu(self.clf1(x))
        out = F.relu(self.clf2(out))
        out = F.relu(self.clf3(out))
        #out = self.sigmoid(out)
        #print('After FillyConn shape:', out.shape)
        return out


#Initializing Model:
clf = Classifier()
print(clf) 

#Cude setup:
cuda_available = torch.cuda.is_available()
cuda_available

if cuda_available:
    clf = clf.cuda()

#Setting up loss and optimizer funtions:
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(clf.parameters(), lr=0.001)

#Training and prediction step:
for epoch in range(50):
    losses = []
    train_total = 0
    train_correct = 0
    # Train
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = clf(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.data)
        #Training Accuracy:
        _, train_predicted = torch.max(outputs.data, 1)
        train_total += targets.size(0)
        train_correct += train_predicted.eq(targets.data).cpu().sum()

    print('Epoch : %d Loss : %.3f Training Acc : %.3f' % (epoch, torch.mean(torch.stack(losses)), 100.*train_correct/train_total))

# Evaluate
clf.eval()
result = []
loc = []
i = 0
for batch_idx, (inputs, targets) in enumerate(testloader):
    if cuda_available:
        inputs, targets = inputs.cuda(), targets.cuda()    
    inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
    outputs = clf(inputs)
    _, predicted = torch.max(outputs.data, 1)
    print(_, predicted)
    result.append(predicted)
    loc.append(testloader.dataset.imgs[i])
    i = i + 1
    
print('--------------------------------------------------------------')

location = []
for i in loc:
  x = i[0].split('/u/bhardwas/Documents/DL/Kaggle_cat_dog/ift6135h19/testset/test/')[1].split('.jpg')[0]
  location.append(x)

import pandas as pd
final_out = pd.DataFrame(torch.stack(result).cpu().numpy())
final_out['location'] = location
final_out.to_csv('/u/bhardwas/Documents/DL/Kaggle_cat_dog/ift6135h19/resultsss.csv')


torch.save(clf.state_dict(), '/u/bhardwas/Documents/DL/Kaggle_cat_dog/model_all')

clf = Classifier()
clf.load_state_dict(torch.load('/u/bhardwas/Documents/DL/Kaggle_cat_dog/model_all'))
clf.cuda()
clf.eval()




#######################################################################################################################################
######################################################### END #########################################################################
#######################################################################################################################################