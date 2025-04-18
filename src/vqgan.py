import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from tqdm import tqdm

from vqvae import VQVAE
from utils import print_progress_bar, CustomImageFolder, deconvolve
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = ".."
epochs = 1000
batch_size = 64
learning_rate = 1e-4
beta = 0.30            # Codebook commitment
gan_weight = 0.7       # Weight of GAN loss
use_perceptual = False # Toggle if you want to try VGG-based perceptual loss
image_size = 256
freeze_epochs = 10

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0)  # No sigmoid (use BCEWithLogits)
        )
    
    def forward(self, x):
        return self.net(x)
    
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return F.l1_loss(x_vgg, y_vgg)
    
    
def train(load=False):
    
    print("Beginning Loading VQGAN")
    vqvae = VQVAE()
    D = Discriminator()
    
    if(load):
        vqvae.load_state_dict(torch.load(f"{root}/models/vae.pth", map_location=device))
        for param in vqvae.parameters():
            param.requires_grad = False
    
    D.to(device)
    vqvae.to(device)
    perceptual_loss_fn = PerceptualLoss() if use_perceptual else None

    # Optimizers
    optimiser = torch.optim.Adam(vqvae.parameters(), lr=learning_rate)
    gan_optimiser = torch.optim.Adam(D.parameters(), lr=learning_rate)
    mse_loss = torch.nn.MSELoss()

    dataset = CustomImageFolder(f"{root}/data/downloaded_images")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    
    print("Beginning training")

    for epoch in range(epochs):
        
        if epoch == freeze_epochs:
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

            recon_loss = mse_loss(recon_images, images)

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
        
        torch.save({'vqgan': vqvae.state_dict(), 'discriminator': D.state_dict()}, f'{root}/models/vqgan_network.pth')


if __name__ == "__main__":
    answer = input("Train existing[1] or new model[2]?: ")
    load = False
    if answer == "1":
        load = True
    train(load)