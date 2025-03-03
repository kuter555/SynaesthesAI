import torch
from torch import nn
from torch.nn import functional


# https://www.geeksforgeeks.org/create-model-using-custom-module-in-pytorch/
class Encoder(nn.Module):
    
    def __init__(self, input_dim=3, latent_dim=256):
        super(Encoder, self).__init__()
    
        self.conv1 = nn.Conv2d(input_dim, 64, 4, stride=2, padding=1)  # 256 -> 128
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # 128 -> 64
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)  # 64 -> 32
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)  # 32 -> 16
        self.conv5 = nn.Conv2d(512, 1024, 4, stride=2, padding=1)  # 16 -> 8
    
    
    def forward(self, x):
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = functional.relu(self.conv3(x))
        x = functional.relu(self.conv4(x))
        x = self.conv5(x)
                
        return x



# https://www.geeksforgeeks.org/create-model-using-custom-module-in-pytorch/
class Decoder(nn.Module):
    
    def __init__(self, output_dim=3, embedding_dim=64, latent_dim=1024):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        flatten_dim= 8 * 8 * self.latent_dim
        
        self.fc = nn.Linear(embedding_dim * 8 * 8, flatten_dim)
        self.tpconv1 = nn.ConvTranspose2d(self.latent_dim, 512, 4, stride=2, padding=1)  # 8 -> 16
        self.tpconv2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)  # 16 -> 32
        self.tpconv3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # 32 -> 64
        self.tpconv4 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 64 -> 128
        self.tpconv5 = nn.ConvTranspose2d(64, output_dim, 4, stride=2, padding=1)
        
        
    def forward(self, x):
        
        batch_size = x.shape[0]  # 64
        x = x.view(batch_size, -1)  # Reshape (64, 64, 8, 8) -> (64, 4096)

        
        x = self.fc(x)
        x = x.view(-1, self.latent_dim, 8, 8)
        x = functional.relu(self.tpconv1(x))
        x = functional.relu(self.tpconv2(x))
        x = functional.relu(self.tpconv3(x))
        x = functional.relu(self.tpconv4(x))
        x = torch.sigmoid(self.tpconv5(x))
        
        return x
    
    
class VQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.beta = beta
        self.embedding_dim = embedding_dim
        
        # Create embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)


    def forward(self, z):
        
        B, _, H, W = z.shape
        z_flat = z.permute(0,2,3,1).contiguous().view(-1, self.embedding_dim)
        
        distances = torch.cdist(z_flat, self.embedding.weight)
        
        encoding_indices = torch.argmin(distances, dim=1)
        
        z_q_flat = self.embedding(encoding_indices)

        z_q = z_q_flat.view(B, H, W, self.embedding_dim).permute(0,3,1,2).contiguous()
        vq_loss = torch.mean((z_q.detach() - z) ** 2)
        commitment_loss = self.beta * torch.mean((z.detach() - z_q) ** 2)


        return z_q, encoding_indices, vq_loss + commitment_loss