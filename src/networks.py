import torch
from torch import nn
from torch.nn import functional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioEncoder(nn.Module):
    
    def __init__(self, in_channels=1, latent_dim=512):
        
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=latent_dim // 2),
            nn.BatchNorm2d(latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=latent_dim // 2, out_channels=latent_dim),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(256, latent_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x