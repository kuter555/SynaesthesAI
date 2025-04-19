import torch
from torch import optim
from PIL import Image
import numpy as np
from vqvae import VQVAE
from utils import print_progress_bar, CustomImageFolder, deconvolve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = ".."
epochs = 250
    
def train_vae(model_name, load=False, image_size=256):
    
    # Pretrained Autoencoder
    model_name = "pae_" + model_name
    
    dataset = CustomImageFolder(f"{root}/data/downloaded_images", image_size=image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    
    model = VQVAE()
    if(load):
        model.load_state_dict(torch.load(f"{root}/models/{model_name}", map_location=device))
    model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = torch.nn.MSELoss()
    
    
    for epoch in range(epochs):
        
        for i, (images, _) in enumerate(dataloader):
            
            print_progress_bar(epoch, i, len(dataloader))
            
            # convert images for pytorch
            images = images.to(device)
        
            # actual training    
            optimiser.zero_grad()
            recon_images, codebook_loss = model(images)    
            recon_loss = mse_loss(recon_images, images)
            loss = recon_loss + codebook_loss
            loss.backward()
            optimiser.step()        

        # save model
        try:
            torch.save(model.state_dict(), f"{root}/models/{model_name}")
        except:
            print(f"\nFailed to save at epoch: {epoch}")
        
        
        if epoch % 25 == 0:
            try:
                torch.save(model.state_dict(), f"{root}/models/BACKUP_{epoch}_{model_name}")
            except:
                print(f"\nFailed to save backup: {epoch}")
        torch.cuda.empty_cache()
    
    

if __name__ == "__main__":
    while True:
        answer = input("Train existing [1] or new model [2]? ").strip()
        if answer in ["1", "2"]:
            load = answer == "1"
            break
        print("Invalid input. Please enter 1 or 2.")

    while True:
        name = input("Please enter the name of your model: ").strip()
        if name:
            if not name.endswith(".pth"):
                name += ".pth"
            break
        print("Model name cannot be empty.")
    
    while True:
        try:
            size = int(input("Please enter the size of your images (max 256): ").strip())
            if 0 < size <= 256:
                break
            else:
                print("Size must be a positive number no greater than 256.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    train_vae(name, load=load, image_size=size) 
