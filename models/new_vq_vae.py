import numpy as np
from models.networks import Encoder, Decoder, VQ

import torch
from torch import nn


class VQVAE(nn.Module):
    
    def __init__(self, 
                 input_dim=3,
                 latent_dim=1024,
                 embedding_dim=128,
                 num_embeddings=512,
                 beta=0.25):
        super(VQVAE, self).__init__()
        
        self.beta = beta
        
        # little bit deep a network but hey ho
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 64, 4, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, latent_dim, 4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        
        self.pre_codebook =  nn.Conv2d(latent_dim, embedding_dim, kernel_size=1)
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.post_codebook = nn.Conv2d(embedding_dim, latent_dim, kernel_size=1)
        
        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(latent_dim, 512, 4, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_dim, 4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        
        
        
        ## rewrite when you have the chance
        x = self.encoder(x)
        quant_input = self.pre_codebook(x)
        
        B, C, H, W = quant_input.shape
        quant_input = quant_input.permute(0,2,3,1)
        quant_input = quant_input.reshape((quant_input.size(0), -1, quant_input.size(-1)))
        
        dist = torch.cdist(quant_input, self.codebook.weight[None, :].repeat((quant_input.size(0), 1, 1)))
        
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        quant_out = torch.index_select(self.codebook.weight, 0, min_encoding_indices.view(-1))
        
        quant_input = quant_input.reshape((-1, quant_input.size(-1)))
        
        commitment_loss = torch.mean((quant_out.detach() - quant_input)**2)
        codebook_loss = torch.mean((quant_out - quant_input.detach()**2))
        quantize_losses = codebook_loss + self.beta * commitment_loss
        
        quant_out = quant_input + (quant_out -  quant_input).detach()
        
        quant_out = quant_out.reshape((B, H, W, C)).permute(0,3,1,2)
        
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        
        decoder_input= self.post_codebook(quant_out)
        output = self.decoder(decoder_input)
        return output, quantize_losses