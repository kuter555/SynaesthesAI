import os
import torch
from models.vqvae import VQVAE
from PIL import Image
from utils import deconvolve, CustomImageFolder
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = "vae.pth"
image = ".test_images/test_im_2_test_.jpeg"


def test_vqvae(model, image):
    
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
        Image.fromarray((deconvolve(recon_images[0].cpu().detach().numpy().squeeze()).transpose(1,2,0) * 255).astype(np.uint8)).save(".test_images/recon.jpeg")


if __name__ == "__main__":
    
    print("Testing novel image reconstruction...")
    test_vqvae(model, image)
    print("Done")