import torch
from torch.nn import functional
from torch import optim

from PIL import Image
import numpy as np
import os

import cv2

#from models.vqvae import VQVAE

from models.new_vq_vae import VQVAE
from utils import print_progress_bar, CustomImageFolder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deconvolve(x):
    return ((x * 0.5) + 0.5).clip(0, 1)


def loss_function(x, x_hat, codebook_loss):
    
    reproduction_loss = functional.binary_cross_entropy(x_hat, x, reduction='sum') # normal loss function
    return reproduction_loss + codebook_loss



def train_vae(epochs=10, load=False):
    
    folder_name = ".outputs/"
    dataset = CustomImageFolder(".downloaded_images")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    
    
    model = VQVAE()
    if(load):
        model.load_state_dict(torch.load("vae.pth", weights_only=True))
    model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        
        stored_figures= []
        for i, (images, _) in enumerate(dataloader):
            
            print_progress_bar(epoch, i, len(dataloader))
            
            # convert images for pytorch
            images = images.to(device)
        
            # actual training    
            optimiser.zero_grad()
            
            recon_images, losses = model(images)
            
            #Image.fromarray((deconvolve(images[0].cpu().numpy().squeeze()).transpose(1,2,0) * 255).astype(np.uint8)).save(".outputs/test.png")
                      
            mse_loss = torch.nn.MSELoss()
            recon_loss = mse_loss(recon_images, deconvolve(images))
            loss = recon_loss + losses         
            
            loss.backward()
            optimiser.step()
            
            # Display images every 500 batches
            if i % 100 == 0:
                with torch.no_grad():
                    stored_figures.append([images[0].cpu().numpy().squeeze(), recon_images[0].cpu().numpy().squeeze()])
            
        try:
            for k in range(len(stored_figures)):
                # Original Image
                orig_img = stored_figures[k][0]
                orig_img = deconvolve(orig_img).transpose(1, 2, 0)  # Convert from CHW to HWC format

                # Convert numpy array to PIL Image and save as PNG
                orig_img_pil = Image.fromarray((orig_img * 255).astype(np.uint8))
                orig_img_pil.save(f"{folder_name}{epoch}_original_{k}.jpg")

                recon_img = stored_figures[k][1]
                recon_img = deconvolve(recon_img).transpose(1, 2, 0)

                recon_img = Image.fromarray((recon_img * 255).astype(np.uint8))
                recon_img.save(f"{folder_name}{epoch}_recon_{k}.jpg")
        except:
            print(f"\nEpoch {i} images failed. Continuing...")
        

        try:
            torch.save(model.state_dict(), "vae.pth")
        except:
            print("\nFailed to save vae")
        
        torch.cuda.empty_cache()
    
if __name__ == "__main__":
    load = input("Load existing model? (y/n): ")
    load = False if load.strip().lower() == "n" else True
    epochs = input("Please enter number of epochs: ")
    try:
        epochs = int(epochs)
    except:
        print("Invalid input - Defaulting to 10 Epochs")
        epochs = 10    
    train_vae(epochs=epochs, load=load)
    input("\nPress Enter to exit...")
