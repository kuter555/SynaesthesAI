import torch
from torch import nn
from torch.nn import functional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    
class old_VQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.5):
        super().__init__()
        self.beta = beta
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        # Create embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)



# mostly copied from https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
# Either Cite properly or rewrite
    def forward(self, z):
        
        B, _, H, W = z.shape
        z_flat = z.permute(0,2,3,1).contiguous().view(-1, self.embedding_dim)
        
        
        # My original
        distances = torch.cdist(z_flat, self.embedding.weight)
        
        # one I found online
        distances = torch.sum(z_flat ** 2, dim = 1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flat, self.embedding.weight.t())
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings).to(device)
        min_encodings.scatter_(1, encoding_indices, 1)
        
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(B, self.embedding_dim, H, W)
        
        
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z.detach() - z_q) ** 2)
        z_q = z + (z_q - z).detach()
        
        
        # adding in a perplexity function?
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        
        z_q = z_q.contiguous()
        
        return z_q, perplexity, encoding_indices, min_encodings, loss
    
    
    
class SonnetExponentialMovingAverage(nn.Module):
    # See: https://github.com/deepmind/sonnet/blob/5cbfdc356962d9b6198d5b63f0826a80acfdf35b/sonnet/src/moving_averages.py#L25.
    # They do *not* use the exponential moving average updates described in Appendix A.1
    # of "Neural Discrete Representation Learning".
    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros(*shape))
        self.register_buffer("average", torch.zeros(*shape))

    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay ** self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average



#https://github.com/airalcorn2/vqvae-pytorch/blob/master/vqvae.py
class VQ(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, use_ema=False, decay=0.25, epsilon=1e-2):
        super().__init__()
        # See Section 3 of "Neural Discrete Representation Learning" and:
        # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L142.

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.use_ema = use_ema
        # Weight for the exponential moving average.
        self.decay = decay
        # Small constant to avoid numerical instability in embedding updates.
        self.epsilon = epsilon

        # Dictionary embeddings.
        limit = 3 ** 0.5
        e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(
            -limit, limit
        )
        if use_ema:
            self.register_buffer("e_i_ts", e_i_ts)
        else:
            self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))

        # Exponential moving average of the cluster counts.
        self.N_i_ts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        # Exponential moving average of the embeddings.
        self.m_i_ts = SonnetExponentialMovingAverage(decay, e_i_ts.shape)

    def forward(self, x):
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1)
        quantized_x = functional.embedding(
            encoding_indices.view(x.shape[0], *x.shape[2:]), self.e_i_ts.transpose(0, 1)
        ).permute(0, 3, 1, 2)

        # See second term of Equation (3).
        if not self.use_ema:
            dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()
        else:
            dictionary_loss = None

        # See third term of Equation (3).
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()
        # Straight-through gradient. See Section 3.2.
        quantized_x = x + (quantized_x - x).detach()

        if self.use_ema and self.training:
            with torch.no_grad():
                # See Appendix A.1 of "Neural Discrete Representation Learning".

                # Cluster counts.
                encoding_one_hots = functional.one_hot(
                    encoding_indices, self.num_embeddings
                ).type(flat_x.dtype)
                n_i_ts = encoding_one_hots.sum(0)
                # Updated exponential moving average of the cluster counts.
                # See Equation (6).
                self.N_i_ts(n_i_ts)

                # Exponential moving average of the embeddings. See Equation (7).
                embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                self.m_i_ts(embed_sums)

                # This is kind of weird.
                # Compare: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L270
                # and Equation (8).
                N_i_ts_sum = self.N_i_ts.average.sum()
                N_i_ts_stable = (
                    (self.N_i_ts.average + self.epsilon)
                    / (N_i_ts_sum + self.num_embeddings * self.epsilon)
                    * N_i_ts_sum
                )
                self.e_i_ts = self.m_i_ts.average / N_i_ts_stable.unsqueeze(0)

        return quantized_x, encoding_indices.view(x.shape[0], -1), dictionary_loss, commitment_loss
