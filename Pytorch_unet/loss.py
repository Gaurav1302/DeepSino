import torch
import torch.nn as nn
import numpy as np

def rms_loss(pred, target, loss_mul):
    return (loss_mul)*(torch.mean((torch.pow(pred - target, 2))))
