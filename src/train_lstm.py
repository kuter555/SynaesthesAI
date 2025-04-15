import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from utils import CustomImageFolder, print_progress_bar
from vqvae import VQVAE, LatentLSTM, InheritedLatentLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "C:/Users/chwah/Dropbox/Family/Christopher/University/Y3/Year Long Project/SynaesthesAI"

class LatentDataset(Dataset):
    def __init__(self, sequences):
        self.data = sequences
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        return sequence[:-1], sequence[1:]


# need this one conditioned off of the top dataset
class InheritedLatentDataset(Dataset):
    def __init__(self, b_sequences, sequences):
        self.b_sequences = b_sequences
        self.sequences = sequences
    
    def __len__(self):
        return len(self.b_sequences)
    
    def __getitem__(self, idx):
        b_sequence = self.b_sequences[idx]
        sequence = self.sequences[idx]
        return sequence, b_sequence[:-1], b_sequence[1:]    
    



def train_lstm(num_epochs=100, load_top=False):

    root = ".."

    torch.cuda.empty_cache()
    model = VQVAE()
    try:
        model.load_state_dict(torch.load(f"{root}/models/vae.pth", map_location=device))
        model.to(device)
    
    except Exception as e:
        print(f"Failed to run: {e} Exiting...")
        return -1
    
    try:
        top_latents = torch.load(f"{root}/models/renewed_top_latents.pt").to(device)
        bottom_latents_1 = torch.load(f"{root}/models/renewed_bottom_latents_1.pt").to(device)    
        
    except Exception as e:
        print(f"Error: {e}\n Failed to load latents. Have you extracted them yet?...\n")
        return -1

    vocab_size = model.num_embeddings    
    top_sequence = top_latents.view(top_latents.size(0), -1)
    bottom_sequence = bottom_latents_1.view(bottom_latents_1.size(0), -1)
    
    t_dataset = LatentDataset(top_sequence)
    b_dataset = InheritedLatentDataset(bottom_sequence, top_sequence)   
    
    lstm = LatentLSTM(vocab_size, model.num_embeddings, hidden_dim=3, layers=3).to(device)
    bottom_lstm = InheritedLatentLSTM(vocab_size, model.num_embeddings, hidden_dim=3, layers=3).to(device)
    criterion = nn.CrossEntropyLoss()    
    optimiser = optim.Adam(lstm.parameters(), lr=1e-3)
    
    t_dataloader = DataLoader(t_dataset, batch_size=32, shuffle=True)
    b_dataloader = DataLoader(b_dataset, batch_size=32, shuffle=True)
    
    if not load_top:
        # train the top lstm
        for epoch in range(num_epochs):

            print_progress_bar("Top LSTM", epoch, num_epochs)
            for batch in t_dataloader:

                inputs, targets = batch
                inputs, targets = inputs.to(device).long(), targets.to(device).long()

                optimiser.zero_grad()
                outputs, _ = lstm(inputs)
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
                loss.backward()
                optimiser.step()

            torch.cuda.empty_cache()

        try:
            torch.save(lstm.state_dict(), f"{root}/models/t_lstm.pth")
        except:
            print("Couldn't save top lstm?")
        
    # train the bottom LSTM
    for epoch in range(num_epochs):
        print_progress_bar("Bottom LSTM", epoch, num_epochs)
        for i, (top_seq, bottom_inputs, bottom_targets) in enumerate(b_dataloader):
            top_seq, bottom_inputs, bottom_targets = (
                top_seq.to(device),
                bottom_inputs.to(device),
                bottom_targets.to(device),
            )

            optimiser.zero_grad()
            outputs, _ = bottom_lstm(bottom_inputs, top_seq)  # Conditioned generation
            loss = criterion(outputs.view(-1, vocab_size), bottom_targets.view(-1))
            loss.backward()
            optimiser.step()
            
        torch.cuda.empty_cache()
            
        try:
            torch.save(bottom_lstm.state_dict(), f"{root}/models/b_lstm.pth")
        except:
            print("\nFailed to save LSTM")


def extract_latent_codes():
    
    print("Beginning Extraction")
    
    dataset = CustomImageFolder("../data/downloaded_images")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    
    model = VQVAE()
    print("Loading Model Dict")
    model.load_state_dict(torch.load("../models/vae.pth", map_location=device))
    model.to(device)
    
    print("Starting image processing")
    
    with torch.no_grad():
        model.eval()
        stored_latent_t = []
        stored_latent_b = []
        for i, (images, _) in enumerate(dataloader):
            
            print_progress_bar("Extracting", i, len(dataloader))
            
            images = images.to(device)
            _, _, _, index_t, index_b = model.encode(images)

            stored_latent_t.append(index_t.cpu())
            stored_latent_b.append(index_b.cpu())
    
    stored_latent_b = torch.cat(stored_latent_b, dim=0)
    stored_latent_t = torch.cat(stored_latent_t, dim=0)
        
    torch.save(stored_latent_t, "../models/renewed_top_latents.pt")
    torch.save(stored_latent_b, "../models/renewed_bottom_latents_1.pt")
            
    print("Latents successfully saved!")
    

if __name__ == "__main__":
    
    answer = input("Would you like to Extract Latents (1) or Train the LSTM (2)? > ")
    if answer == "1":
        extract_latent_codes()
    
    else:
        
        x = input("Do you want to skip training the top (Y)? ")
        if x.lower() == "y":
            train_lstm(load_top=True)
        else:
            train_lstm(load_top=False)
    