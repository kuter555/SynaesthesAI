import numpy as np
from models.networks import Encoder, Decoder, VQ

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
class VQVAE(nn.Module):
    def __init__(self):
        
        super().__init__()      
        
        latent_dim = 1024
        num_embeddings = 512
        embedding_dim = 64
        
        self.encoder = Encoder(latent_dim=latent_dim)
        
        self.pre_codebook = nn.Conv2d(latent_dim, embedding_dim, kernel_size=1, stride=1)
        
        self.codebook = VQ(num_embeddings, embedding_dim)
        
        self.decoder = Decoder(embedding_dim=embedding_dim, latent_dim=latent_dim)
                
                
    def forward(self, x):
        z_e = self.pre_codebook(self.encoder(x))
        z_q, perplexity, _, _, loss = self.codebook(z_e)
        x_hat = self.decoder(z_q)
        
        return x_hat, loss, perplexity