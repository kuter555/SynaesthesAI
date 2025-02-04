import torch
from torch import nn
from torch.nn import functional
from torch import optim

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from vqvae import VAE
import matplotlib.pyplot as plt

import sys


def print_progress_bar(epoch, iteration, total, length=50):
    progress = int(length * iteration / total)
    bar = f"\033[31m Epoch {epoch}:\033[97m [{'=' * progress}{' ' * (length - progress)}] {int(100 * iteration / total)}%"
    sys.stdout.write(f"\r{bar}")
    sys.stdout.flush()
    
    


def train_vae(load=False):
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    dataset = datasets.CIFAR100(root=r"./.cifar100-data", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = VAE()
    if(load):
        model.load_state_dict(torch.load("vae", weights_only=True))
    
    optimiser = optim.Adam(model.parameters(), lr=1e-4)
          
    
    for epoch in range(5):
        
        stored_figures= []  
        for i, (images, _) in enumerate(dataloader):
            
            print_progress_bar(epoch, i, len(dataloader))
            
            # convert images for pytorch
            images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            model.to(images.device)
        
            # actual training    
            optimiser.zero_grad()
            
            recon_images = model(images)
            recon_loss = functional.mse_loss(recon_images, images)
            recon_loss.backward()
            optimiser.step()
            
            # Display images every 500 batches
            if i % 500 == 0:
                with torch.no_grad():

                    stored_figures.append([images[0].cpu().numpy().squeeze(), recon_images[0].cpu().numpy().squeeze()])
            
            
        # display results
        fig, axes = plt.subplots(len(stored_figures), 2, figsize=(6, 3 * len(stored_figures)))
        for k in range(len(stored_figures)):
                        
            # Original Image
            orig_img = stored_figures[k][0]
            orig_img = orig_img.transpose(1,2,0)
            axes[k][0].imshow(orig_img)
            axes[k][0].set_title("Original")
            axes[k][0].axis("off")


            # Reconstructed Image
            recon_img = stored_figures[k][1]
            recon_img = recon_img.transpose(1,2,0)
            axes[k][1].imshow(recon_img)
            axes[k][1].set_title("Reconstructed")
            axes[k][1].axis("off")
            
        plt.tight_layout()
        plt.savefig(rf".images/output_{epoch}.png", bbox_inches="tight")
        plt.close(fig)
            
    torch.save(model.state_dict(), "vae")
    
if __name__ == "__main__":
    train_vae(False)
    input("\nPress Enter to exit...")