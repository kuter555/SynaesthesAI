import torch
import torch.nn.functional as F

from train_vqvae import train_vae
from networks import VQVAE2, Discriminator, PerceptualLoss
from utils import print_progress_bar, CustomImageFolder
    
import os
from os import getenv
from dotenv import load_dotenv

load_dotenv()
root = getenv('root')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 1000
batch_size = 64
learning_rate = 1e-4
beta = 0.30            # Codebook commitment
gan_weight = 0.7       # Weight of GAN loss
use_perceptual = False # Toggle if you want to try VGG-based perceptual loss
image_size = 256
freeze_epochs = 10


    
    
def train(model_name, vqvae_model="", load=False, image_size=256):


    vqvae = VQVAE2()
    D = Discriminator()
    
    if vqvae_model != "":
        vqvae.load_state_dict(torch.load(os.path.join(f"{root}/models/{model_name}"), map_location=device))
        for param in vqvae.parameters():
            param.requires_grad = False

    
    D.to(device)
    vqvae.to(device)
    perceptual_loss_fn = PerceptualLoss() if use_perceptual else None

    # Optimizers
    optimiser = torch.optim.Adam(vqvae.parameters(), lr=learning_rate)
    gan_optimiser = torch.optim.Adam(D.parameters(), lr=learning_rate)

    dataset = CustomImageFolder(f"{root}/data/downloaded_images", image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    
    print("Beginning training")

    for epoch in range(epochs):
        
        if load and epoch == freeze_epochs:
            print(f"Unfreezing VQ-VAE at epoch {epoch}")
            for param in vqvae.parameters():
                param.requires_grad = True
                
        for i, (images, _) in enumerate(dataloader):
            
            print_progress_bar(epoch, i, len(dataloader))
                        
            # convert images for pytorch
            images = images.to(device)
            
            # actual training
            recon_images, codebook_loss = vqvae(images)

            # gan training
            gan_real = D(images)
            gan_fake = D(recon_images.detach())
            gan_loss = F.binary_cross_entropy_with_logits(gan_real, torch.ones_like(gan_real)) + \
                     F.binary_cross_entropy_with_logits(gan_fake, torch.zeros_like(gan_fake))

            gan_optimiser.zero_grad()
            gan_loss.backward()
            gan_optimiser.step()

            gan_loss = F.binary_cross_entropy_with_logits(D(recon_images), torch.ones_like(gan_fake))
            recon_loss = F.l1_loss(recon_images, images)

            if use_perceptual:
                recon_loss += perceptual_loss_fn(recon_images, images)

            total_loss = recon_loss + beta * codebook_loss + gan_weight * gan_loss

            optimiser.zero_grad()
            total_loss.backward()
            optimiser.step()
        
        # save model
        try:
            torch.save({'vqgan': vqvae.state_dict(), 'discriminator': D.state_dict()}, f'{root}/models/{model_name}')
        except:
            print(f"\nFailed to save at epoch: {epoch}")
            
        # incremental "backup" save in case anything else fails
        if epoch % 25 == 0:
            try:
                torch.save({'vqgan': vqvae.state_dict(), 'discriminator': D.state_dict()}, f'{root}/models/BACKUP_{epoch}_{model_name}')
            except:
                print(f"\nFailed to save backup: {epoch}")
            
        
        torch.cuda.empty_cache()
    
            
            

if __name__ == "__main__":

    og_model = ""
    while True:
        answer = input("Train existing [1] or new model [2]? ").strip()
        if answer == "1":
            load = True
            break
        elif answer == "2":
            load = False
            load_vqvae = input("Would you like to use a premade VQVAE? (y/n): ")
            if load_vqvae == "y":
                og_model = input("What is the name of your premade VQVAE?: ")
                break
            else:
                break            
        else:
            print("Invalid input. Please enter 1 or 2.")

    while True:
        name = input("Please enter the name of your VQGAN model: ").strip()
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


    train(name, vqvae_model=og_model, load=load, image_size=size)