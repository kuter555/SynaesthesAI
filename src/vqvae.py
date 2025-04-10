import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import custom_distributed as dist_fn

# WOAH: https://github.com/BhanuPrakashPebbeti/Image-Generation-Using-VQVAE/blob/main/vqvae-gpt.ipynb

# WOAH: https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch

class LatentLSTM(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_dim, layers):
        super(LatentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_dim)
        
    def forward(self, x, h=None):
        
        print("INPUT SHAPE:", x.shape)
        print("INPUT DTYPE:", x.dtype)
        print("INPUT MIN/MAX:", x.min().item(), x.max().item())
        
        x = self.embedding(x)
        out, h = self.lstm(x, h)
        out = self.fc(out)
        return out, h
      
        
class InheritedLatentLSTM(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_dim, layers):
        super(InheritedLatentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_dim)
        
    def forward(self, x, y, h=None):
        x = self.embedding(x)
        y = self.embedding(y)
        full_input = torch.cat((x, y), dim=-1)
        out, h = self.lstm(full_input, h)
        out = self.fc(out)
        return out, h
        


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, input_dim, channels, n_residual_blocks, n_residual_dims, stride):
        super().__init__()    
        network = []
        
        if stride==4:
            network.extend([nn.Conv2d(input_dim, channels // 2, 4, stride=2, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(channels // 2, channels, 4, stride=2, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(channels, channels, 3, padding=1),])
        
        elif stride==2:
            network.extend([nn.Conv2d(input_dim, channels // 2, 4, stride=2, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(channels // 2, channels, 3, padding=1)])
        
        network.extend([ResBlock(channels, n_residual_dims) for _ in range(n_residual_blocks)])
        network.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*network)
                
    
    def forward(self, x):
        x = self.model(x)
        return x
        
        
class Decoder(nn.Module):
    
    def __init__(self, input_dim, output_dim, channels, n_residual_blocks, n_residual_dims, stride):
        
        super().__init__()
        
        network = [nn.Conv2d(input_dim, channels, 3, padding=1),
            *[ResBlock(channels, n_residual_dims) for _ in range(n_residual_blocks)],
            nn.ReLU(inplace=True)]
        
        if stride == 2:
            network.append(nn.ConvTranspose2d(channels, output_dim, 4, stride=2, padding=1))
        
        elif stride==4:
            network.extend([nn.ConvTranspose2d(channels, channels // 2, 4, stride=2, padding=1), 
                            nn.ReLU(inplace=True), 
                            nn.ConvTranspose2d(channels//2, output_dim, 4, stride=2, padding=1)])
        
        self.model = nn.Sequential(*network)            
        
    
    def forward(self, x):
        x = self.model(x)
        return x


class VQVAE(nn.Module):
    
    def __init__(self, 
                 input_dim=3,
                 channels=128,
                 
                 n_residual_blocks=2,
                 n_residual_dims=32,
                 
                 embedding_dim=64,
                 num_embeddings=512,
                 
                 beta=0.25):
        super(VQVAE, self).__init__()
        
        self.num_embeddings = num_embeddings
        
        self.beta = beta
        
        self.bottom_encoder = Encoder(input_dim, channels, n_residual_blocks, n_residual_dims, stride=4)
        self.top_encoder = Encoder(channels, channels, n_residual_blocks, n_residual_dims, stride=2)

        self.pre_codebook_top = nn.Conv2d(channels, embedding_dim, 1)
        self.codebook_top = Quantize(embedding_dim, num_embeddings)
        self.post_codebook_top = nn.ConvTranspose2d(embedding_dim, embedding_dim, 4, stride=2, padding=1)
        
        self.pre_codebook_bottom = nn.Conv2d(embedding_dim + channels, embedding_dim, 1)
        self.codebook_bottom = Quantize(embedding_dim, num_embeddings)        
        
        self.top_decoder = Decoder(embedding_dim, embedding_dim, channels, n_residual_blocks, n_residual_dims, stride=2)
        self.bottom_decoder = Decoder(embedding_dim + embedding_dim, input_dim, channels, n_residual_blocks, n_residual_dims, stride=4)
        
        
    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff.mean() * self.beta


    def encode(self, input):
        
        enc_b = self.bottom_encoder(input)
        enc_t = self.top_encoder(enc_b)

        quant_t = self.pre_codebook_top(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.codebook_top(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.top_decoder(quant_t)
        
        print(f"Enc_b shape: {enc_b.shape}")
        print(f"Dec_t shape: {dec_t.shape}")
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.pre_codebook_bottom(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.codebook_bottom(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        # quant is the quantised values, id is the indices
        return quant_t, quant_b, diff_t + diff_b, id_t, id_b



    def decode(self, quant_t, quant_b):
        upsample_t = self.post_codebook_top(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.bottom_decoder(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.codebook_top.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.codebook_bottom.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec