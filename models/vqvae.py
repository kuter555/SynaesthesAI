import numpy as np
from models.networks import Encoder, Decoder

from torch import nn
from torch import optim

# May need to refactor
class VAE(nn.Module):
    def __init__(self):
        
        super().__init__()        
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x