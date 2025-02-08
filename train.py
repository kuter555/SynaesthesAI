import torch
from torch import nn
from torch.nn import functional
from torch import optim

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

from vqvae import VAE
from utils import print_progress_bar

class CustomImageFolder(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = []
        
        self.transform =  transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Supported image extensions
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        # Collect all valid image files
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.valid_extensions):
                    self.image_files.append(os.path.join(root, file))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")  # Open image and ensure it's RGB
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # We don't need labels for VAE (return dummy label 0)







def train_vae(load=False):
    
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    #dataset = datasets.CIFAR100(root=r"./.cifar100-data", train=True, transform=transform, download=True)
    #dataloader = DataLoader(dataset, batch_size=32, shuffle=True)    
    dataset = CustomImageFolder(".downloaded_images")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
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
            if i % 100 == 0:
                with torch.no_grad():
                    stored_figures.append([images[0].cpu().numpy().squeeze(), recon_images[0].cpu().numpy().squeeze()])
            
            
        for k in range(len(stored_figures)):
            # Original Image
            orig_img = stored_figures[k][0]
            orig_img = orig_img.transpose(1, 2, 0)  # Convert from CHW to HWC format
    
            # Convert numpy array to PIL Image and save as PNG
            orig_img_pil = Image.fromarray((orig_img * 255).astype(np.uint8))  # Assuming pixel values are normalized between 0 and 1
            orig_img_pil.save(f"{epoch}_original_{k}.png")
            
            recon_img = stored_figures[k][1]
            recon_img = recon_img.transpose(1, 2, 0)  # Convert from CHW to HWC format
    
            # Convert numpy array to PIL Image and save as PNG
            recon_img = Image.fromarray((recon_img * 255).astype(np.uint8))  # Assuming pixel values are normalized between 0 and 1
            recon_img.save(f"{epoch}_recon_{k}.png")
            
            
    torch.save(model.state_dict(), "vae")
    
    
if __name__ == "__main__":
    train_vae(True)
    input("\nPress Enter to exit...")