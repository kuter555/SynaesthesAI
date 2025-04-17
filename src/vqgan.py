import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from tqdm import tqdm

from vqvae import VQVAE
from utils import print_progress_bar, CustomImageFolder, deconvolve
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = ".."
epochs = 100
batch_size = 32
learning_rate = 2e-4
beta = 0.25            # Codebook commitment
gan_weight = 0.8       # Weight of GAN loss
use_perceptual = False # Toggle if you want to try VGG-based perceptual loss
image_size = 256


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
    
    
def train(load=True):    
    vqvae = VQVAE().to(device)
    D = Discriminator().to(device)
    perceptual_loss_fn = PerceptualLoss() if use_perceptual else None

    # Optimizers
    g_opt = torch.optim.Adam(vqvae.parameters(), lr=learning_rate)
    d_opt = torch.optim.Adam(D.parameters(), lr=learning_rate)

    dataset = CustomImageFolder(f"{root}/data/downloaded_images")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    
    model = VQVAE()
    if(load):
        model.load_state_dict(torch.load(f"{root}/models/vae.pth", map_location=device))
    model.to(device)

    for epoch in range(epochs):
        
        print_progress_bar("", epoch, epochs)
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for real_images in loop:
            real_images = real_images.to(device)

            recon_images, vq_loss = vqvae(real_images)

            D_real = D(real_images)
            D_fake = D(recon_images.detach())
            d_loss = F.binary_cross_entropy_with_logits(D_real, torch.ones_like(D_real)) + \
                     F.binary_cross_entropy_with_logits(D_fake, torch.zeros_like(D_fake))

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            gan_loss = F.binary_cross_entropy_with_logits(D(recon_images), torch.ones_like(D_fake))
            recon_loss = F.l1_loss(recon_images, real_images)

            if use_perceptual:
                recon_loss += perceptual_loss_fn(recon_images, real_images)

            total_loss = recon_loss + beta * vq_loss + gan_weight * gan_loss

            g_opt.zero_grad()
            total_loss.backward()
            g_opt.step()

            loop.set_postfix({
                "d_loss": d_loss.item(),
                "vq_loss": vq_loss.item(),
                "recon": recon_loss.item(),
                "gan": gan_loss.item()
            })

        torch.save(vqvae.state_dict(), f"{root}/models/vqgan.pth")


if __name__ == "__main__":
    train()