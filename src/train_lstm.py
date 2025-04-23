import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image

from utils import print_progress_bar, deconvolve, AudioLatentDataset, AudioInheritedLatentDataset, LatentDataset, InheritedLatentDataset, extract_latent_codes, extract_audio_latent_codes
from networks import VQVAE, VQVAE2, LatentLSTM, InheritedLatentLSTM, AudioLatentLSTM, AudioInheritedLatentLSTM

from dotenv import load_dotenv
from os import getenv, makedirs 
from os.path import join, exists

load_dotenv()
root = getenv('root')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Train LSTM with loss against audio

# need to input the mel-spectrograms of the audio


# ALLOWS TRAINING OF VQVAE-2 and VQGAN
def train_audio_lstm_hierarchical(model_name, t_latents, b_latents, size, num_epochs=100, load_top=False):

    answer = input("Do you need to generate latent codes? (y/n): ")
    if answer == "y":
        
        output_path = input("What is your desired output path?: ")
        if not exists(join(root, "models", "LSTM", output_path)):
            makedirs(join(root, "models", "LSTM", output_path))
        
        extract_audio_latent_codes(model_name, t_latents, b_latents, size, output_path)

    torch.cuda.empty_cache()
    model = VQVAE2()
    try:
        model_path = join(root, "models", model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
        t_path = join(root, "models/LSTM", output_path, t_latents)
        b_path = join(root, "models/LSTM", output_path, b_latents)
        top_latents = torch.load(t_path).to(device)
        bottom_latents = torch.load(b_path).to(device)
        
        audio_path = join(root, "models", output_path, model_name.split(".")[0] + "_audio.pth")
        audio_info = torch.load(audio_path).to(device)

    except Exception as e:
        print(f"Failed to run: {e} Exiting...")
        return -1
    
    if not exists(join(root, "models", "LSTM", model_name.split(".")[0])):
        makedirs(join(root, "models", "LSTM", model_name.split(".")[0]))

    vocab_size = model.num_embeddings    
    top_sequence = top_latents.view(top_latents.size(0), -1)
    bottom_sequence = bottom_latents.view(bottom_latents.size(0), -1)
    
    t_dataset = AudioLatentDataset(top_sequence, audio_info)
    b_dataset = AudioInheritedLatentDataset(bottom_sequence, top_sequence, audio_info)   
    
    lstm = AudioLatentLSTM(vocab_size, model.num_embeddings, hidden_dim=3, layers=3, audio_dim=audio_info.shape[1], audio_embed_dim=512).to(device)
    bottom_lstm = AudioInheritedLatentLSTM(vocab_size, model.num_embeddings, hidden_dim=3, layers=3, audio_dim=audio_info.shape[1], audio_embed_dim=512).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(lstm.parameters(), lr=1e-3)
    
    t_dataloader = DataLoader(t_dataset, batch_size=32, shuffle=True)
    b_dataloader = DataLoader(b_dataset, batch_size=32, shuffle=True)
    
    if not load_top:
        for epoch in range(num_epochs):
            for i, (inputs, target, audio) in enumerate(t_dataloader):
                print_progress_bar(epoch, i, len(t_dataloader))

                # Get items
                inputs, target, audio = inputs.to(device).long(), target.to(device).long(), audio.to(device).long()

                # Train LSTM
                optimiser.zero_grad()
                outputs, _ = lstm(inputs, audio)
                loss = criterion(outputs.view(-1, vocab_size), target.view(-1))
                loss.backward()
                optimiser.step()

            torch.cuda.empty_cache()
            try:
                torch.save(lstm.state_dict(), join(root, "models", "LSTM", model_name.split(".")[0], "t_lstm.pth"))
            except:
                print("Couldn't save top LSTM")
            
            
    # train the bottom LSTM
    for epoch in range(num_epochs):
        for i, (top_seq, bottom_inputs, bottom_targets, audio) in enumerate(b_dataloader):
            print_progress_bar(epoch, i, len(b_dataloader))
            top_seq, bottom_inputs, bottom_targets, audio = (
                top_seq.to(device),
                bottom_inputs.to(device),
                bottom_targets.to(device),
                audio.to(device)
            )

            optimiser.zero_grad()
            outputs, _ = bottom_lstm(bottom_inputs, top_seq, audio)
            loss = criterion(outputs.view(-1, vocab_size), bottom_targets.view(-1))
            loss.backward()
            optimiser.step()
            
        torch.cuda.empty_cache()
        try:
            torch.save(bottom_lstm.state_dict(), join(root, "models", "LSTM", model_name.split(".")[0], "b_lstm.pth"))
        except:
            print("\nFailed to save bottom LSTM")







# ALLOWS TRAINING OF VQVAE-2 and VQGAN
def train_vanilla_lstm_hierarchical(model_name, t_latents, b_latents, num_epochs=100, load_top=False):

    torch.cuda.empty_cache()
    model = VQVAE2()
    try:
        model_path = join(root, "models", model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
        t_path = join(root, "models", t_latents)
        b_path = join(root, "models", b_latents)
        top_latents = torch.load(t_path).to(device)
        bottom_latents = torch.load(b_path).to(device)    

    except Exception as e:
        print(f"Failed to run: {e} Exiting...")
        return -1
    
    if not exists(join(root, "models", "LSTM", model_name.split(".")[0])):
        makedirs(join(root, "models", "LSTM", model_name.split(".")[0]))

    vocab_size = model.num_embeddings    
    top_sequence = top_latents.view(top_latents.size(0), -1)
    bottom_sequence = bottom_latents.view(bottom_latents.size(0), -1)
    
    t_dataset = LatentDataset(top_sequence)
    b_dataset = InheritedLatentDataset(bottom_sequence, top_sequence)   
    
    lstm = LatentLSTM(vocab_size, model.num_embeddings, hidden_dim=3, layers=3).to(device)
    bottom_lstm = InheritedLatentLSTM(vocab_size, model.num_embeddings, hidden_dim=3, layers=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(lstm.parameters(), lr=1e-3)
    
    t_dataloader = DataLoader(t_dataset, batch_size=32, shuffle=True)
    b_dataloader = DataLoader(b_dataset, batch_size=32, shuffle=True)
    
    if not load_top:
        for epoch in range(num_epochs):
            for i, (inputs, target) in enumerate(t_dataloader):
                print_progress_bar(epoch, i, len(t_dataloader))

                # Get items
                inputs, target = inputs.to(device).long(), target.to(device).long()

                # Train LSTM
                optimiser.zero_grad()
                outputs, _ = lstm(inputs)
                loss = criterion(outputs.view(-1, vocab_size), target.view(-1))
                loss.backward()
                optimiser.step()

            torch.cuda.empty_cache()
            try:
                torch.save(lstm.state_dict(), join(root, "models", "LSTM", model_name.split(".")[0], "t_lstm.pth"))
            except:
                print("Couldn't save top LSTM")
            
    # train the bottom LSTM
    for epoch in range(num_epochs):
        for i, (top_seq, bottom_inputs, bottom_targets) in enumerate(b_dataloader):
            print_progress_bar(epoch, i, len(b_dataloader))
            top_seq, bottom_inputs, bottom_targets = (
                top_seq.to(device),
                bottom_inputs.to(device),
                bottom_targets.to(device),
            )

            optimiser.zero_grad()
            outputs, _ = bottom_lstm(bottom_inputs, top_seq)
            loss = criterion(outputs.view(-1, vocab_size), bottom_targets.view(-1))
            loss.backward()
            optimiser.step()
            
        torch.cuda.empty_cache()
        try:
            torch.save(bottom_lstm.state_dict(), join(root, "models", "LSTM", model_name.split(".")[0], "b_lstm.pth"))
        except:
            print("\nFailed to save bottom LSTM")


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
    
    while True:
        model_name = input("Please enter the name of your model: ").strip()
        if model_name:
            if not model_name.endswith(".pth"):
                model_name += ".pth"
            break
        print("Model name cannot be empty.")

    while True:
        t_latents = input("What is the name of your top latents?: ").strip()
        if t_latents:
            if not t_latents.endswith(".pth"):
                t_latents += ".pt"
            break
        print("Top latents cannot be empty.")
        
    while True:
        b_latents = input("What is the name of your bottom latents?: ").strip()
        if b_latents:
            if not b_latents.endswith(".pth"):
                b_latents += ".pt"
            break
        print("Bottom latents cannot be empty.")
    
    
    while True:
        try:
            size = int(input("Enter the size of your images (max 256): ").strip())
            if 0 < size <= 256:
                break
            else:
                print("Number must be between 1 and 256.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")
    
    
    while True:
        try:
            num_epochs = int(input("Enter the number of training epochs (max 2000): ").strip())
            if 0 < num_epochs <= 2000:
                break
            else:
                print("Number must be between 1 and 2000.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")
            
    
    while True:
        answer = input("Would you like to Extract Latents (1) or Train the LSTM (2), or Decode (3)? > ")
        if answer == "1":
            output_path = input("What is your desired output path?: ")
            extract_latent_codes(model_name, t_latents, b_latents, size, output_path)
            break
        
        elif answer == "2":
            while True:
                Load = input("Load existing top [1] or new top [2]? ").strip()
                if Load in ["1", "2"]:
                    Load = answer == "1"
                    break
                print("Invalid input. Please enter 1 or 2.")
                
            while True:
                model_type = input("What model? VAE [1], or VQVAE2/VQGAN-FT [2], or non-audio [3]?: ").strip()
                if model_type == "1":
                    break
                elif model_type == "2":
                    train_audio_lstm_hierarchical(model_name, t_latents, b_latents, size, num_epochs, Load)
                elif model_type == "3":
                    train_vanilla_lstm_hierarchical(model_name, t_latents, b_latents, num_epochs, Load)
                    break
                else:
                    print("Invalid input. Please enter 1, 2, or 3.")            
            break
        
        elif answer == "3":
            break
        
        else:
            print("Answer must be one of 1, 2, or 3.")