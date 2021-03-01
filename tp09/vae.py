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

from models import VAE, ConvVAE

SAVE_DIR_CHECKPOINT = Path("advanced_dl/checkpoints")
SAVE_DIR_IMAGES = Path("advanced_dl/results")

def load_mnist_dataset(train=True):
    return dset.MNIST(
        root='.advanced_dl/data/MNIST',
        train=train,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

def generate(vae_model, inp, from_latent=True, path=None):
    """
    inp: input, either a batch of latent variable if from_latent=True
        or a batch of images to reconstruct
    """
    if from_latent:
        examples = vae_model.decoder(inp).detach().cpu()
    else:
        examples = vae_model(inp)[0].detach().cpu()
    examples = examples.view(-1, 1, 28, 28)
    img = vutils.make_grid(examples, padding=0, normalize=True, nrow=inp.size(0))
    if from_latent:
        plt.figure(figsize=(15,15))
        plt.axis("off")
        plt.title(f"Generated Images")
        plt.imshow(np.transpose(img.cpu(),(1,2,0)))
    else:
        orig = vutils.make_grid(inp.view(-1, 1, 28, 28), padding=0, normalize=True, nrow=inp.size(0))
        grid = torch.empty(size=(3, examples.size(-1)*2, examples.size(0)*examples.size(-2)))
        grid[:, :examples.size(-2), :] = orig
        grid[:, examples.size(-2):, :] = img
        plt.figure(figsize=(30,15))
        plt.axis("off")
        plt.title(f"Reconstructed Images")
        plt.imshow(np.transpose(grid.cpu(),(1,2,0)))

    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
        plt.close()


def train(vae_model, optimizer, loaders, writer, args):
    train_loader, test_loader = loaders
    n_batches = len(train_loader)
    cross_entropy = nn.BCELoss(reduction='mean')
    device = torch.device(args.device)

    # To track the evolution of generated images from a single batch of noises
    fixed_z = torch.randn(64, args.latent_dim, device=device)

    for epoch in range(args.num_epochs):
        epoch_train_loss = 0
        epoch_test_loss = 0
        losses = {'loss': {"train": 0,'test': 0 },
                  'reconstruction_loss': {"train": 0,'test': 0 },
                  'kl': {"train": 0,'test': 0 } }
        start = time.time()

        for i, batch in enumerate(train_loader):
            samples, _ = batch
            samples = samples.to(device)
            sample_size = samples.size(0)

            if args.architecture == "linear":
                samples = samples.flatten(1)
            reconstructed, mu, logsigma = vae_model(samples)

            reconstruction_loss = cross_entropy(reconstructed, samples)
            kl_losses = (- logsigma + mu**2 + torch.exp(logsigma) - 1) / 2
            kl = kl_losses.sum()
            loss = reconstruction_loss + kl
            
            # Track losses
            losses["loss"]["train"] += loss
            losses["reconstruction_loss"]["train"] += reconstruction_loss
            losses["kl"]["train"] += kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%100 == 0:
                print(f"Epoch {epoch+1}/{args.num_epochs} - Batch {i+1}/{n_batches}: \ Train Loss: {loss.item() :.6}")

        for key in losses:
            losses[key]["train"] /= len(train_loader)

        generate(vae_model, samples[-10:, :], from_latent=False,
                path=args.save_path_img / f"train_{epoch}.png")
        
        for i, batch in enumerate(test_loader):
            samples, _ = batch
            samples = samples.to(device)

            if args.architecture == "linear":
                samples = samples.flatten(1)
            reconstructed, mu, logsigma = vae_model(samples)

            reconstruction_loss = cross_entropy(reconstructed, samples)
            kl_losses = - (1 + logsigma - mu**2 - torch.exp(logsigma)).sum(-1) / 2
            kl = kl_losses.mean()
            loss = reconstruction_loss + kl

            # Track losses
            losses["loss"]["test"] += loss
            losses["reconstruction_loss"]["test"] += reconstruction_loss
            losses["kl"]["test"] += kl

        for key in losses:
            losses[key]["test"] /= len(test_loader)

        duration = (time.time() - start) / 60
        print("\n\n", "*"*100)

        print(f"Epoch {epoch+1}/{args.num_epochs} - (Duration: {duration :.2} min): \
                Train Loss: {losses['loss']['train'] :.4}, \
                Test Loss: {losses['loss']['test'] :.4}")
        print("*"*100, "\n\n")

        # Save loss & accuracy
        writer.add_scalars('Loss', losses["loss"], epoch)
        writer.add_scalars('Reconstruction_Loss', losses["reconstruction_loss"], epoch)
        writer.add_scalars('KL', losses["kl"], epoch)
        
        generate(vae_model, samples[-10:, :], from_latent=False,
                path=args.save_path_img / f"test_{epoch}.png")

        # Save Model
        with (args.save_path_ckpt / f"model_latent{args.latent_dim}").open("wb") as f:
            torch.save(vae_model, f)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train a Variational Auto Encoder')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
    parser.add_argument('--latent_dim', default=3, type=int, help='Dimension of the latent space')
    parser.add_argument('--architecture', default="conv", choices=["linear", "conv"],
                        type=str, help='Type of architecture: linear or conv')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning Rate')
    parser.add_argument('--num_epochs', default=15, type=int, help='Number of epochs')
    parser.add_argument('--device', default="cuda:0", type=str, help='Device')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    train_set = load_mnist_dataset(train=True)
    test_set = load_mnist_dataset(train=False)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=args.batch_size, drop_last=True)

    for architecture in ["conv", "linear"]:
        for latent_dim in [2, 3, 5, 10, 50]:
            args.architecture = architecture
            args.latent_dim = latent_dim

            outdir = Path(f'advanced_dl/runs/{args.architecture}_{args.latent_dim }_vae')
            writer = SummaryWriter(outdir / str(int(time.time())))
            args.save_path_ckpt = SAVE_DIR_CHECKPOINT / f"{args.architecture}_vae"
            args.save_path_img = SAVE_DIR_IMAGES / f"{args.architecture}_{args.latent_dim}vaeMNIST"
            if not args.save_path_ckpt.is_dir():
                args.save_path_ckpt.mkdir()
            if not args.save_path_img.is_dir():
                args.save_path_img.mkdir()
            
            if args.architecture == "linear":
                vae_model = VAE(latent_dim=args.latent_dim, original_dim=784, hidden_dim=256).to(device)
            else:
                vae_model = ConvVAE(latent_dim=args.latent_dim, ngf=32, nc=1).to(device)
            
            optimizer = optim.Adam(vae_model.parameters(), lr=args.lr)
            train(vae_model, optimizer, (train_loader, test_loader), writer, args)

    