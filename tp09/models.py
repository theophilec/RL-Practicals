import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn    
import torch.optim as optim


SAVE_DIR_CHECKPOINT = Path("advanced_dl/checkpoints/linear_vae")
SAVE_DIR_IMAGES = Path("advanced_dl/results/vaeMNIST")

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fcmu = nn.Linear(hidden_dim, latent_dim)
        self.fcsigma = nn.Linear(hidden_dim, latent_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        x = self.actvn(self.fc1(x))
        mu = self.fcmu(x)
        logsigma = self.fcsigma(x)
        return mu, logsigma

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.actvn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.actvn(self.fc1(z))
        x = self.sigmoid(self.fc2(x))
        return x

class ConvEncoder(nn.Module):
    def __init__(self, nc, ngf, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(nc, ngf, 4, 2, 1)
        self.convmu = nn.Conv2d(ngf, ngf, 4, 2, 1)
        self.convsigma = nn.Conv2d(ngf, ngf, 4, 2, 1)
        self.fcmu = nn.Linear(ngf*7*7, latent_dim)
        self.fcsigma = nn.Linear(ngf*7*7, latent_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        x = self.actvn(self.conv1(x))
        mu = self.actvn(self.convmu(x)).flatten(1)
        mu = self.fcmu(mu)
        logsigma = self.actvn(self.convsigma(x)).flatten(1)
        logsigma = self.fcsigma(logsigma)
        return mu, logsigma

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, ngf, nc):
        super().__init__()
        self.convt1 = nn.ConvTranspose2d(latent_dim, ngf * 4, 4, 1, 0)
        self.convt2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1)
        self.convt3 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 2)
        self.convt4 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1)
        self.actvn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = z[:, :, None, None]
        x = self.actvn(self.convt1(z))
        x = self.actvn(self.convt2(x))
        x = self.actvn(self.convt3(x))
        x = self.sigmoid(self.convt4(x))
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim, original_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(input_dim=original_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=original_dim)

    def sample_z(self, mu, logsigma):
        """Samples n realizations of the latent variable Z with the reparametrization trick.
            mu: (batch_size x latent_dim)
            logsima: (batch_size x latent_dim)
        """
        epsilon = torch.randn(mu.shape, device=mu.device)
        samples = epsilon * torch.exp(logsigma / 2) + mu
        return samples

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        z = self.sample_z(mu, logsigma)
        return self.decoder(z), mu, logsigma

class ConvVAE(nn.Module):
    def __init__(self, latent_dim, ngf, nc=1):
        super().__init__()
        self.encoder = ConvEncoder(nc=nc, ngf=ngf, latent_dim=latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim, ngf=ngf, nc=nc)

    def sample_z(self, mu, logsigma):
        """Samples n realizations of the latent variable Z with the reparametrization trick.
            mu: (batch_size x latent_dim)
            logsima: (batch_size x latent_dim)
        """
        epsilon = torch.randn(mu.shape, device=mu.device)
        samples = epsilon * torch.exp(logsigma / 2) + mu
        return samples

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        z = self.sample_z(mu, logsigma)
        return self.decoder(z), mu, logsigma
