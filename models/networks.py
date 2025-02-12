import torch
from torch import nn
from torch.nn import functional


# https://www.geeksforgeeks.org/create-model-using-custom-module-in-pytorch/
class Encoder(nn.Module):
    
    def __init__(self, input_dim=3, latent_dim=8):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=4, stride=2, padding=1)  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(32, latent_dim, kernel_size=4, stride=2, padding=1)  # 14x14 -> 7x7
    
    def forward(self, x):
        x = functional.relu(self.conv1(x))
        x = self.conv2(x)
                
        return x



# https://www.geeksforgeeks.org/create-model-using-custom-module-in-pytorch/
class Decoder(nn.Module):
    
    def __init__(self, output_dim=3, latent_dim=64):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        flatten_dim=latent_dim*64*64
        
        self.fc = nn.Linear(latent_dim, flatten_dim)
        self.tpconv1 = nn.ConvTranspose2d(latent_dim, 32, kernel_size=4, stride=2, padding=1)
        self.tpconv2 = nn.ConvTranspose2d(32, output_dim, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.latent_dim, 64, 64)
        x = functional.relu(self.tpconv1(x))
        x = torch.sigmoid(self.tpconv2(x))
        
        return x