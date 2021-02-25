from collections import OrderedDict
from collections import namedtuple
from itertools import product

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=1)       
        
    def forward(self, t):
        # (1) input layer
        t = t
        
        #(2) latent space vector => hidden dimension
        t = t.reshape(t.size(0), -1)
        t = self.fc1(t)
        t = F.relu(t)
        
        #(3) hidden linear layer
        t = self.fc2(t)
        t = torch.sigmoid(t)
        
        return t


class Generator(nn.Module):
    
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=latent_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)       
        
    def forward(self, t):
        # (1) input layer
        t = t
        
        #(2) latent space vector => hidden dimension
        t = self.fc1(t)
        t = F.relu(t)
        
        #(3) hidden linear layer
        t = self.fc2(t)
        t = torch.sigmoid(t)
        
        return t