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


def train_gpt(model, top_latents, bottom_latents, num_epochs=100):
    
    root = ".."
    print("Pre training")
    torch.cuda.empty_cache()
    vqvae = VQVAE()    
    try:
        t_latents = torch.load(f"{root}/models/{top_latents}").to(device)
        b_latents = torch.load(f"{root}/models/{bottom_latents}").to(device)
        vqvae.load_state_dict(torch.load(f"{root}/models/{model}", map_location=device))
        vqvae.to(device)
    except Exception as e:
        try:
            checkpoint = torch.load(f"{root}/models/vqgan-128.pth", map_location=device)
            vqvae.load_state_dict(checkpoint["vqgan"])
        except Exception as e:
            print(f"Unable to load model: {e}. Exiting...")
        
    
    print("Loaded latents")
    top_sequence = t_latents.view(t_latents.size(0), -1)
    bottom_sequence = b_latents.view(b_latents.size(0), -1)
    
    t_dataset = LatentDataset(top_sequence)
    b_dataset = HierarchicalLatentDataset(top_sequence, bottom_sequence)   
    
    t_dataloader = DataLoader(t_dataset, batch_size=32, shuffle=True)
    b_dataloader = DataLoader(b_dataset, batch_size=32, shuffle=True)
    
    t_model = LatentGPT(vocab_size=vqvae.num_embeddings).to(device)
    b_model = BottomGPT(vocab_size=vqvae.num_embeddings).to(device)
    t_optimiser = torch.optim.Adam(t_model.parameters(), lr=1e-4)
    b_optimiser = torch.optim.Adam(b_model.parameters(), lr=1e-4)
    
    print("Established optimsers and datasets")
    
    print("Training top GPT...")
    for epoch in range(num_epochs):
        for i, (input_ids, target_ids) in enumerate(t_dataloader):
            
            print_progress_bar(epoch, i, len(b_dataloader))
            
            input_ids = input_ids.to(device)       
            target_ids = target_ids.to(device)     

            logits = t_model(input_ids)            

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )

            loss.backward()
            t_optimiser.step()
            t_optimiser.zero_grad()
        try:
            torch.save(t_model.state_dict(), f"{root}/models/t_gpt.pth")
        except Exception as e:
            print(f"Couldn't save top GPT: {e}")
    
    
    print("Training bottom GPT...")
    for epoch in range(num_epochs):
        for i, (input_ids, target_ids) in enumerate(b_dataloader):
            
            print_progress_bar(epoch, i, len(b_dataloader))
            
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
    
    top_latents = input("What is the name of your top latents?: ")
    bottom_latents = input("What is the name of your bottom latents?: ")
    model = input("What is the name of your model?: ")
    
    train_gpt(model, top_latents, bottom_latents)