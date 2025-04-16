from vqvae import LatentGPT
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from utils import print_progress_bar, HierarchicalLatentDataset, LatentDataset
from vqvae import VQVAE, LatentGPT, BottomGPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "C:/Users/chwah/Dropbox/Family/Christopher/University/Y3/Year Long Project/SynaesthesAI"


def train_gpt(num_epochs=100):
    
    root = ".."
    
    torch.cuda.empty_cache()
    vqvae = VQVAE()    
    try:
        vqvae.load_state_dict(torch.load(f"{root}/models/vae.pth", map_location=device))
        vqvae.to(device)
        top_latents = torch.load(f"{root}/models/renewed_top_latents.pt").to(device)
        bottom_latents_1 = torch.load(f"{root}/models/renewed_bottom_latents_1.pt").to(device)   
    except Exception as e:
        print(f"Failed to run: {e} Exiting...")
        return -1
    
    top_sequence = top_latents.view(top_latents.size(0), -1)
    bottom_sequence = bottom_latents_1.view(bottom_latents_1.size(0), -1)
    
    t_dataset = LatentDataset(top_sequence)
    b_dataset = HierarchicalLatentDataset(top_sequence, bottom_sequence)   
    
    t_dataloader = DataLoader(t_dataset, batch_size=32, shuffle=True)
    b_dataloader = DataLoader(b_dataset, batch_size=32, shuffle=True)
    
    t_model = LatentGPT(vocab_size=vqvae.num_embeddings).to(device)
    b_model = BottomGPT(vocab_size=vqvae.num_embeddings).to(device)
    t_optimiser = torch.optim.Adam(t_model.parameters(), lr=1e-4)
    b_optimiser = torch.optim.Adam(b_model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        
        print_progress_bar("Top GPT", epoch, num_epochs)
        for input_ids, target_ids in t_dataloader:
            input_ids = input_ids.to(device)       # shape: (batch, 1023)
            target_ids = target_ids.to(device)     # shape: (batch, 1023)

            logits = t_model(input_ids)              # shape: (batch, 1023, vocab_size)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # flatten logits
                target_ids.view(-1)
            )

            loss.backward()
            t_optimiser.step()
            t_optimiser.zero_grad()
    
        
        try:
            torch.save(t_model.state_dict(), f"{root}/models/t_gpt.pth")
        except:
            print("Couldn't save top lstm?")
    
    for epoch in range(num_epochs):
        
        print_progress_bar("Bottom GPT", epoch, num_epochs)
        for input_ids, target_ids in b_dataloader:
            input_ids = input_ids.to(device)       # (batch, 5120)
            target_ids = target_ids.to(device)     # (batch, 5120)

            logits = b_model(input_ids)              # (batch, 5120, vocab_size)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=-100  # don't learn from top tokens
            )

            loss.backward()
            b_optimiser.step()
            b_optimiser.zero_grad()
        
        
        try:
            torch.save(b_model.state_dict(), f"{root}/models/b_gpt.pth")
        except:
            print("Couldn't save bottom lstm?")


if __name__ == "__main__":
    train_gpt()