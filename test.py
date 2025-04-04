import os
import torch
from models.vqvae import VQVAE
from PIL import Image
from utils import deconvolve, CustomImageFolder
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = "vae.pth"

def test_vqvae(model):
    
    if not os.path.exists(model):    
        print("Model not found.")
        return
    
    dataset = CustomImageFolder(".test_images")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

    model = VQVAE()
    model.to(device)
    model.load_state_dict(torch.load("vae.pth", weights_only=True, map_location=device))
    

    for i, (images, _) in enumerate(dataloader):
        
        images = images.to(device)
        recon_images, _ = model(images)
        for i in range(len(recon_images)):        
            Image.fromarray((deconvolve(recon_images[i].cpu().detach().numpy().squeeze()).transpose(1,2,0) * 255).astype(np.uint8)).save(f".test_images/recon_{i}.jpeg")


if __name__ == "__main__":
    
    print("Testing novel image reconstruction...")
    test_vqvae(model)
    print("Done")