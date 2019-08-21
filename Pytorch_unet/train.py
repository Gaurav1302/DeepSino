import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import pickle
from Load_Data import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from unet1 import *
from collections import defaultdict
import torch.nn.functional as F
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from loss import *
from torchsummary import summary
CUDA_DEVICE = 'cuda:0'
SAVE_PATH = './weights/weights_train6_2-1000epochs.pth'
print('\n' + SAVE_PATH + '\n')

GAMMA = 0.98
MOMENTUM = 0.9
STEP_SIZE = 1
LR = 1e-3
NUM_EPOCHS =  250    
LOSS_MUL = 1e4
num_patches = 100
n_samples = 1000
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 1
TRANSFORM = torchvision.transforms.ToTensor()
BATCH_SIZE  = 128
START_FILTERS = 32 # Starting with these many filters in u-net 
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

x, y = train_dataset.__getitem__(43)

# Make batches and iterate over these batches
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)
train_iter = iter(train_loader)
val_iter = iter(val_loader)


print('#'*26)
print('# Completed Data Loading #')
print('#'*26)


image_datasets = {
    'train': train_dataset, 
    'val': val_dataset
}

dataloaders = {
    'train': train_loader,
    'val': val_loader
}


device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else 'cpu')
model = UNet(f = START_FILTERS)# start_filter = START_FILTERS
model = model.to(device)
summary(model, input_size=(IMG_CHANNELS, IMG_WIDTH, IMG_HEIGHT))
exit()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    loss = rms_loss(pred, target, loss_mul = LOSS_MUL)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss



def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs))) 



def train_model(model, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("Learning Rate: ", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, SAVE_PATH)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


print('#'*15)
print('# Train Model #')
print('#'*15)

optimizer_ft = optim.Adam(model.parameters(), lr=LR)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=STEP_SIZE, gamma=GAMMA)        
        
model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

torch.save(model.state_dict(), SAVE_PATH)


# Testing the model over numerical and experimental data stored in 'Test_data/' 
print('#'*11)
print('# Testing #')
print('#'*11)

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn as nn
import scipy.io as sio
import os
from loss import *
from unet1 import *

IMG_CHANNELS = 1
IMG_HEIGHT= 512                                                                                    
IMG_WIDTH = 512
WEIGHTS_PATH = SAVE_PATH 
TEST_PATH = './Test_data/'
SAVE_PATH = './Results/' + (WEIGHTS_PATH.split('_')[1].split('.')[0])
lst = os.listdir(TEST_PATH)
n_samples = len(lst)//2

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
    print('\nFolder Created', SAVE_PATH)
else:
    print('\nAlready exists', SAVE_PATH)




# Load Model
device = torch.device('cpu')
model.eval()


X_test = np.zeros((n_samples, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype='float32')
Y_test = np.zeros((n_samples, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype='float32')
full = 'original'
limited = 'limited_noise_interpolated'
files = os.listdir(TEST_PATH)

for file in files:
    if (file.startswith('original')):
        continue
    load_path = TEST_PATH + file
    print(file)
    mat = sio.loadmat(load_path)
    test_img = np.asarray(mat[limited], dtype='float32')
    test_img = np.pad(test_img, [(156,156),(0,0) ], mode = 'constant', constant_values=0)
    plt.imsave('in.png', test_img, cmap='Greys')
    test_img = test_img.reshape(1, 1, IMG_HEIGHT, IMG_WIDTH)
    test_img = torch.Tensor(test_img).cuda()
    print('\n')


    with torch.no_grad():
        model_out = Variable(test_img) 
        model_out = model(model_out)

    pred_out = np.asarray(model_out.cpu().detach().numpy(), dtype = 'float32').reshape(512, 512)
    plt.imsave('pred.png', pred_out, cmap='Greys')
    pred_out = pred_out[156:356, :]
    save_name = file.split('.')[0] + '_pred.mat'
    sio.savemat(SAVE_PATH + '/' + save_name, {"pred_pad": pred_out})
    print('Saving mat file...', save_name)
    print(pred_out)
