import torch
from torch import optim
from PIL import Image
import numpy as np
from src.vqvae import VQVAE
from src.utils import print_progress_bar, CustomImageFolder, deconvolve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_vae(epochs=10, load=False):
    
    folder_name = "data/outputs/"
    dataset = CustomImageFolder("data/downloaded_images")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    
    model = VQVAE()
    if(load):
        model.load_state_dict(torch.load("models/vae.pth", weights_only=True, map_location=device))
    model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        
        stored_figures= []
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
            torch.save(model.state_dict(), "models/vae.pth")
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
