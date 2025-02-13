import numpy as np
from models.networks import Encoder, Decoder

import torch
from torch import nn
from torch import optim

# https://medium.com/@judyyes10/generate-images-using-variational-autoencoder-vae-4d429d9bdb5
#def sampling(args):
#    """Reparameterization trick by sampling from an isotropic unit Gaussian.
#    # Arguments
#        args (tensor): mean and log of variance of Q(z|X)
#    # Returns
#        z (tensor): sampled latent vector
#    """
#
#    z_mean, z_log_var = args
#    batch = K.shape(z_mean)[0]
#    dim = K.int_shape(z_mean)[1]
#    # by default, random_normal has mean = 0 and std = 1.0
#    epsilon = K.random_normal(shape=(batch, dim))
#    return z_mean + K.exp(0.5 * z_log_var) * epsilon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# May need to refactor
class VAE(nn.Module):
    def __init__(self):
        
        super().__init__()      
        
        latent_dim = 256
        flatten_dim = 1024 * 8 * 8
        
        self.encoder = Encoder(latent_dim=latent_dim)
        
        self.mean = nn.Linear(flatten_dim, latent_dim)
        self.logvar = nn.Linear(flatten_dim, latent_dim)
        
        self.decoder = Decoder(latent_dim=latent_dim)
        
    def encode(self, x):
        x = torch.flatten(self.encoder(x), start_dim=1)
        mean = self.mean(x)
        var = self.logvar(x)
        return mean, var
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var*epsilon
        return z
    
    def forward(self, x):
        mean, var = self.encode(x)
        z = self.reparameterization(mean, var)        
        x = self.decoder(z)
        return x, mean, var