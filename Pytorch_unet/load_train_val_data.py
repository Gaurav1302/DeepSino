import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import pickle
from Load_Data import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


num_patches = 100
n_samples = 1000
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 1
TRANSFORM = torchvision.transforms.ToTensor()
BATCH_SIZE  = 32
TRAIN_PATH_X = './data/X_train.pkl'
TRAIN_PATH_Y = './data/Y_train.pkl'

VAL_PATH_X = './data/X_val.pkl'
VAL_PATH_Y = './data/Y_val.pkl'

# Load Dataset
print('\n')
print('#'*35)
print('# Load Training & Validation Data #')
print('#'*35)
print('Train:')
train_dataset = Dataset_sino(TRAIN_PATH_X, TRAIN_PATH_Y,transform = TRANSFORM)


print('Validation:')
val_dataset = Dataset_sino(VAL_PATH_X, VAL_PATH_Y,transform = TRANSFORM)

# x, y = val_dataset.__getitem__(1)
# print(x.shape)
# print(type(x))

# Make batches and iterate over these batches
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)
train_iter = iter(train_loader)
val_iter = iter(val_loader)
# print(type(train_iter))

# Iterate through batches using next()
# images, labels = train_iter.next()
# images_v, labels_v = val_iter.next()
# check if the size of the batch formed is correct
# print('images shape on batch size = {}'.format(images_v.size()))
# print('labels shape on batch size = {}'.format(labels_v.size()))

print('#'*26)
print('# Completed Data Loading #')
print('#'*26)


