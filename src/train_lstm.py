import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import CustomImageFolder, print_progress_bar, deconvolve, LatentDataset, InheritedLatentDataset
from vqvae import VQVAE, LatentLSTM, InheritedLatentLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#root = "C:/Users/chwah/Dropbox/Family/Christopher/University/Y3/Year Long Project/SynaesthesAI"

root = ".."

def train_lstm(num_epochs=100, load_top=False):

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


def extract_latent_codes(model_path, t_latent_name, b_latent_name, image_size, output_path):
    
    print("Beginning Extraction")
    
    dataset = CustomImageFolder(f"{root}/data/downloaded_images", image_size=image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    
    model = VQVAE()
    print("Loading Model Dict")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["vqgan"])
        except Exception as e:
            print(f"Unable to load model: {e}. Exiting...")
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
        
    torch.save(stored_latent_t, f"{root}/{output_path}/{t_latent_name}")
    torch.save(stored_latent_b, f"{root}/{output_path}/{b_latent_name}")
            
    print("Latents successfully saved!")
    
    
def decode():
    
    torch.cuda.empty_cache()
    model = VQVAE()
    try:
        model.load_state_dict(torch.load(f"{root}/models/vae.pth", map_location=device))
        model.to(device)       
        
        top_lstm = LatentLSTM(model.num_embeddings, model.num_embeddings, hidden_dim=3, layers=3).to(device)
        top_lstm.load_state_dict(torch.load(f"{root}/models/t_lstm.pth", map_location=device))

        bottom_lstm = InheritedLatentLSTM(model.num_embeddings, model.num_embeddings, hidden_dim=3, layers=3).to(device)
        bottom_lstm.load_state_dict(torch.load(f"{root}/models/b_lstm.pth", map_location=device))
    
    except Exception as e:
        print(f"Failed to run: {e} Exiting...")
        return -1
    
    top_latents, bottom_latents = sample_latents(top_lstm, bottom_lstm, 0, 65536)
    
    top_latents = top_latents.view(1, 32, 32)
    bottom_latents = bottom_latents.view(1, 64, 64)
    with torch.no_grad():
        recon = model.decode_code(top_latents, bottom_latents)
        Image.fromarray((deconvolve(recon[0].cpu().detach().numpy().squeeze()).transpose(1,2,0) * 255).astype(np.uint8)).save(f"{root}/data/test_images/lstm_test.jpeg")
    
def sample_latents(lstm, bottom_lstm, start_token, _, temperature=0.5):
    
    lstm.eval()
    bottom_lstm.eval()
    
    top_sequence_length = 32 * 32
    bottom_sequence_length = 64 * 64
    
    try:
        top_latents = torch.load(f"{root}/models/renewed_top_latents.pt").to(device)
        top_sequence = top_latents.view(top_latents.size(0), -1)
    except Exception as e:
        print(f"Error: {e}, couldn't extract latents...")
        return -1, -1

    t_dataset = LatentDataset(top_sequence)    
    start_token, _ = t_dataset[0]
    start_token = start_token[0].item()

    top_generated = [start_token]
    input_seq = torch.tensor([[start_token]], device=device).long()
    hidden = None

    for _ in range(top_sequence_length - 1):
        with torch.no_grad():
            output, hidden = lstm(input_seq, hidden)
            top_generated.append(next_token.item())
            input_seq = next_token

    top_generated = torch.tensor(top_generated, device=device).unsqueeze(0).long()  # shape: (1, sequence_length)
    bottom_generated = []
    input_seq = torch.zeros((1, 1), device=device).long()  # Start with some neutral token
    hidden = None

    for _ in range(bottom_sequence_length):  # 64 * 64 = 4096
        with torch.no_grad():
            output, hidden = bottom_lstm(input_seq, top_generated, hidden)
            probs = torch.softmax(output[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            bottom_generated.append(next_token.item())
            input_seq = next_token.unsqueeze(0)
    
    return top_generated, torch.tensor(bottom_generated, device=device)
    
    

if __name__ == "__main__":
    
    answer = input("Would you like to Extract Latents (1) or Train the LSTM (2), or Decode (3)? > ")
    if answer == "1":
        file = input("What is the name of your model?: ")
        path = input("What is the desired output path?: ")
        b_name = input("What is the desired b_latent name: ")
        t_name = input("What is the desired t_latent_name: ")
        extract_latent_codes(f"{root}/models/{file}", t_name, b_name, path)
    
    elif answer == "3":
        decode()
    
    else:
        
        x = input("Do you want to skip training the top (Y)? ")
        if x.lower() == "y":
            train_lstm(load_top=True)
        else:
            train_lstm(load_top=False)
    