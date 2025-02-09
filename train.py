import torch
from torch.nn import functional
from torch import optim

from PIL import Image
import numpy as np

from models.vqvae import VAE
from utils import print_progress_bar, CustomImageFolder


def deconvolve(x):
    
    return ((x * 0.5) + 0.5).clip(0, 1)
    


def loss_function(x, x_hat, mean, log_var):
    
    # get the losses
    reproduction_loss = functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD



def train_vae(load=False):
    
    folder_name = ".outputs/"
    dataset = CustomImageFolder(".downloaded_images")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    model = VAE()
    if(load):
        model.load_state_dict(torch.load("vae", weights_only=True))
    
    optimiser = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(10):
        
        stored_figures= []  
        for i, (images, _) in enumerate(dataloader):
            
            print_progress_bar(epoch, i, len(dataloader))
            
            # convert images for pytorch
            images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            model.to(images.device)
        
            # actual training    
            optimiser.zero_grad()
            
            recon_images, mean, var = model(images)
            recon_loss = loss_function(deconvolve(images), recon_images, mean, var)
            recon_loss.backward()
            optimiser.step()
            
            # Display images every 500 batches
            if i % 100 == 0:
                with torch.no_grad():
                    stored_figures.append([images[0].cpu().numpy().squeeze(), recon_images[0].cpu().numpy().squeeze()])
            
            
        for k in range(len(stored_figures)):
            # Original Image
            orig_img = stored_figures[k][0]
            orig_img = deconvolve(orig_img).transpose(1, 2, 0)  # Convert from CHW to HWC format
    
            # Convert numpy array to PIL Image and save as PNG
            orig_img_pil = Image.fromarray((orig_img * 255).astype(np.uint8))  # Assuming pixel values are normalized between 0 and 1
            orig_img_pil.save(f"{folder_name}{epoch}_original_{k}.png")
            
            recon_img = stored_figures[k][1]
            recon_img = deconvolve(recon_img).transpose(1, 2, 0)  # Convert from CHW to HWC format
    
            # Convert numpy array to PIL Image and save as PNG
            recon_img = Image.fromarray((recon_img * 255).astype(np.uint8))  # Assuming pixel values are normalized between 0 and 1
            recon_img.save(f"{folder_name}{epoch}_recon_{k}.png")
            
        torch.save(model.state_dict(), "vae")
    
    
if __name__ == "__main__":
    train_vae(load=False)
    input("\nPress Enter to exit...")