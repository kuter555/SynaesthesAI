import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg16
from transformers import GPT2Config, GPT2LMHeadModel

import custom_distributed as dist_fn

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

class LatentLSTM(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_dim, layers):
        super(LatentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_dim)
        
    def forward(self, x, h=None):        
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
        
        repeat_factor = x.size(1) // y.size(1) + 1  # Ensures it's long enough
        y_repeated = y.repeat_interleave(repeat_factor, dim=1)  # [32, ~4096, 512]
        y_resized = y_repeated[:, :x.size(1), :]  
        
        if x.dim() > 3:
            x = x.squeeze(2)       
               
        full_input = torch.cat((x, y_resized), dim=-1)  # [32, 4095, 1024]
                
        out, h = self.lstm(full_input, h)
        out = self.fc(out)
        return out, h
    
    

class AudioLatentLSTM(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_dim, layers, audio_dim, audio_embed_dim):
        super(AudioLatentLSTM, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_dim)
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(3 * 256 * 256, audio_embed_dim),
            nn.ReLU(),
            nn.Linear(audio_embed_dim, layers * hidden_dim * 2)
        )
        
    def forward(self, x, audio_features): 
        
        audio_features = audio_features.view(audio_features.size(0), -1)
        x = self.embedding(x)
        
        # GPT generated
        audio_embed = self.audio_encoder(audio_features)
        audio_embed = audio_embed.view(-1, self.layers, self.hidden_dim, 2)
        audio_embed = audio_embed.permute(1, 3, 0, 2)
        h_0, c_0 = audio_embed[:,0], audio_embed[:,1]
        h_0 = h_0.contiguous()
        c_0 = c_0.contiguous()
        
        out, h = self.lstm(x, (h_0, c_0))
        out = self.fc(out)
        return out, h
      
      
class AudioInheritedLatentLSTM(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_dim, layers, audio_dim, audio_embed_dim):
        super(AudioInheritedLatentLSTM, self).__init__()
        
        self.layers = layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_dim)
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(3 * 256 * 256, audio_embed_dim),
            nn.ReLU(),
            nn.Linear(audio_embed_dim, layers * hidden_dim * 2)
        )
        
    def forward(self, x, y, audio_features):
        
        audio_features = audio_features.view(audio_features.size(0), -1)
        x = self.embedding(x)
        y = self.embedding(y)
        repeat_factor = x.size(1) // y.size(1) + 1  # Ensures it's long enough
        y_repeated = y.repeat_interleave(repeat_factor, dim=1)  # [32, ~4096, 512]
        y_resized = y_repeated[:, :x.size(1), :]  
        
        if x.dim() > 3:
            x = x.squeeze(2)       
               
        full_input = torch.cat((x, y_resized), dim=-1)  # [32, 4095, 1024]
                
                
        # GPT generated
        audio_embed = self.audio_encoder(audio_features)
        audio_embed = audio_embed.view(-1, self.layers, self.hidden_dim, 2)
        audio_embed = audio_embed.permute(1, 3, 0, 2)
        h_0, c_0 = audio_embed[:,0], audio_embed[:,1]
        h_0 = h_0.contiguous()
        c_0 = c_0.contiguous()
        
        out, h = self.lstm(full_input, (h_0, c_0))
        out = self.fc(out)
        return out, h


        
class LatentGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, n_layer=6, n_head=8):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=4096,  # max length you expect (top: 1024, bottom: 4096)
            n_ctx=4096,
            n_embd=embedding_dim,
            n_layer=n_layer,
            n_head=n_head,
            use_cache=False,
        )
        self.transformer = GPT2LMHeadModel(config)

    def forward(self, input_ids):
        return self.transformer(input_ids).logits

class BottomGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, n_layer=12, n_head=8):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=5120,  # enough room for [top | bottom]
            n_ctx=5120,
            n_embd=embedding_dim,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.model = GPT2LMHeadModel(config)

    def forward(self, input_ids):
        return self.model(input_ids).logits


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
    
    
    
class EncoderVAE(nn.Module):
    
    def __init__(self, input_dim=3, latent_dim=8):
        super(EncoderVAE, self).__init__()
    
        self.conv1 = nn.Conv2d(input_dim, 64, 4, stride=2, padding=1)  # 256 -> 128
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # 128 -> 64
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)  # 64 -> 32
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)  # 32 -> 16
        self.conv5 = nn.Conv2d(512, 1024, 4, stride=2, padding=1)  # 16 -> 8
    
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
                
        return x



# https://www.geeksforgeeks.org/create-model-using-custom-module-in-pytorch/
class DecoderVAE(nn.Module):
    
    def __init__(self, output_dim=3, latent_dim=64):
        super(DecoderVAE, self).__init__()
        
        self.depth = 1024
        self.latent_dim = latent_dim
        flatten_dim= 8 * 8 * self.depth
        
        self.tpconv1 = nn.ConvTranspose2d(self.depth, 512, 4, stride=2, padding=1)  # 8 -> 16
        self.tpconv2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)  # 16 -> 32
        self.tpconv3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # 32 -> 64
        self.tpconv4 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 64 -> 128
        self.tpconv5 = nn.ConvTranspose2d(64, output_dim, 4, stride=2, padding=1)
        self.fc = None
        
    def forward(self, x, encoded_shape, output_shape):
        B, C, H, W = encoded_shape
        x = self.fc(x)
        x = x.view(B, C, H, W)
        x = F.relu(self.tpconv1(x))
        x = F.relu(self.tpconv2(x))
        x = F.relu(self.tpconv3(x))
        x = F.relu(self.tpconv4(x))
        x = torch.sigmoid(self.tpconv5(x))
        
        x = F.interpolate(x, size=output_shape[2:], mode='bilinear', align_corners=False)
        
        return x

class VAE(nn.Module):
    def __init__(self):
        
        super().__init__()      
        
        self.latent_dim = 256
        self.flatten_dim = 1024 * 8 * 8
        
        self.encoder = EncoderVAE(latent_dim=self.latent_dim)
        
        self.mean = None
        self.logvar = None
        
        self.decoder = DecoderVAE(latent_dim=self.latent_dim)
        
    def encode(self, x):
        
        x = self.encoder(x)
        self.encoded_shape = x.shape
        B, C, H, W = self.encoded_shape

        flatten_dim = C * H * W
        if self.mean is None or self.logvar is None:
            self.mean = nn.Linear(flatten_dim, self.latent_dim).to(x.device)
            self.logvar = nn.Linear(flatten_dim, self.latent_dim).to(x.device)
            self.decoder.fc = nn.Linear(self.latent_dim, flatten_dim).to(x.device)

        x = x.view(B, -1)
        mean = self.mean(x)
        var = self.logvar(x)
        return mean, var
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var*epsilon
        return z
    
    def forward(self, x):
        original_shape = x.shape
        mean, var = self.encode(x)
        z = self.reparameterization(mean, var)        
        x = self.decoder(z, self.encoded_shape, original_shape)
        return x, mean, var


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



class VQVAE2(nn.Module):
    
    def __init__(self, 
                 input_dim=3,
                 channels=128,
                 
                 n_residual_blocks=2,
                 n_residual_dims=32,
                 
                 embedding_dim=64,
                 num_embeddings=512,
                 
                 beta=0.25):
        super(VQVAE2, self).__init__()
       
        
        self.input_dim = input_dim
        self.channels = channels
        self.n_residual_blocks = n_residual_blocks
        self.n_residual_dims = n_residual_dims
        self.embedding_dim = embedding_dim
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
        quant_t, quant_b, diff, _, _, = self.encode(input)
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