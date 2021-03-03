import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
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

from models import VAE, ConvVAE
from vae import generate, load_mnist_dataset

device = torch.device("cuda:0")
path_to_ckpt = Path("advanced_dl/checkpoints/conv_vae/model_latent2")
path_to_img = Path("advanced_dl/results/explore/")

def plot_latent_grid():
    model = torch.load(path_to_ckpt)
    interval = np.arange(-2, 3)
    examples = []
    for z1 in interval:
        for z2 in interval:
            z = torch.tensor([z1, z2], device=device, dtype=torch.float)[None, :]
            torch.manual_seed(0)
            example = model.decoder(z).detach().cpu()
            examples.append(example.view(1, 1, 28, 28))
    examples = torch.cat(examples, 0)
    img = vutils.make_grid(examples, padding=2, normalize=True, nrow=len(interval))
    plt.figure(figsize=(15,15))
    plt.axis("off")
    plt.title(f"Generated Images: Exploration of the latent space of the mean")
    plt.imshow(np.transpose(img.cpu(),(1,2,0)))
    plt.savefig(path_to_img / "grid_latent_space.png")


def num_params(model):
    nump = 0
    for param in model.parameters():
        nump += np.prod(param.shape)
    return nump

def latent_embedding():
    model = torch.load(path_to_ckpt)
    test_set = load_mnist_dataset(train=False)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=500)
    samples, label = next(iter(test_loader))
    samples = samples.to(device)
    mu, logsigma = model.encoder(samples)
    mu = mu.detach().cpu()
    plt.figure(figsize=(15,15))
    plt.title(f"Latent Space Encoding: the 2D case")
    sns.scatterplot(mu[:,0], mu[:,1], hue=label, s=100, palette="Paired")
    plt.legend()
    plt.savefig(path_to_img / "latent_embedding.png")


for architecture in ["linear", "conv"]:
    for latent_dim in [2,3,5,10,50]:
        path_to_ckpt_ = Path(f"advanced_dl/checkpoints/{architecture}_vae/model_latent{latent_dim}")
        model = torch.load(path_to_ckpt_)
        print(architecture, latent_dim)
        print(num_params(model))


# for samples, label in test_loader:
#     samples = samples.to(device)
#     mu, logsigma = model.encoder(samples)

