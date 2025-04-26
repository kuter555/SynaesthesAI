import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image

from utils import print_progress_bar, deconvolve, AudioLatentDataset, AudioInheritedLatentDataset, LatentDataset, CustomAudioImagePairing, InheritedLatentDataset, extract_latent_codes, extract_audio_latent_codes, extract_audio_latent_codes_vae
from networks import VAE, VQVAE, VQVAE2, LatentLSTM, InheritedLatentLSTM, AudioLatentLSTM, AudioInheritedLatentLSTM

from dotenv import load_dotenv
from os import getenv, makedirs 
from os.path import join, exists

load_dotenv()
root = getenv('root')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# trains a LSTM on VAE
def train_audio_lstm_vae(model_name, latents, size, num_epochs=100):
    
    output_path = input("What is your desired output path/where are latents stored?: ")
    answer = input("Do you need to generate latent codes? (y/n): ")
    if answer == "y":
        if not exists(join(root, "models", "LSTM", output_path)):
            makedirs(join(root, "models", "LSTM", output_path))    
        extract_audio_latent_codes_vae(model_name, t_latents, b_latents, size, output_path)

    torch.cuda.empty_cache()
    model = VAE()
    try:
        path = join(root, "models/LSTM", output_path, latents)
        latents = torch.load(path).to(device)
        audio_path = join(root, "models/LSTM", output_path, "audio.pt")
        audio_info = torch.load(audio_path).to(device)
        
    except Exception as e:
        print(f"Failed to load latents: {e} Exiting...")
        return -1
    
    try:
        model_path = join(root, "models", model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))   
    except Exception as e:
        print(f"Unable to load model: {e}. Exiting...")
    model.to(device)     

    if not exists(join(root, "models", "LSTM", model_name.split(".")[0])):
        makedirs(join(root, "models", "LSTM", model_name.split(".")[0]))

    vocab_size = model.num_embeddings    
    sequence = latents.view(latents.size(0), -1)
    
    dataset = AudioLatentDataset(sequence, audio_info)  
    
    lstm = AudioLatentLSTM(vocab_size, model.num_embeddings, hidden_dim=3, layers=3, audio_dim=audio_info.shape[1], audio_embed_dim=512).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(lstm.parameters(), lr=1e-3)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(num_epochs):
        for i, (inputs, target, audio) in enumerate(dataloader):
            print_progress_bar(epoch, i, len(dataloader))
            
            # Get items
            inputs, target, audio = inputs.to(device).long(), target.to(device).long(), audio.to(device)

            # Train LSTM
            optimiser.zero_grad()
            outputs, _ = lstm(inputs, audio)
            loss = criterion(outputs.view(-1, vocab_size), target.view(-1))
            loss.backward()
            optimiser.step()
            
        torch.cuda.empty_cache()
        try:
            torch.save(lstm.state_dict(), join(root, "models", "LSTM", output_path, "lstm.pth"))
        except:
            print("Couldn't save LSTM")
            



# ALLOWS TRAINING OF VQVAE-2 and VQGAN
def train_audio_lstm_hierarchical(model_name, t_latents, b_latents, size, num_epochs=100, load_top=False):

    output_path = input("What is your desired output path/where are latents stored?: ")
    answer = input("Do you need to generate latent codes? (y/n): ")
    if answer == "y":
        if not exists(join(root, "models", "LSTM", output_path)):
            makedirs(join(root, "models", "LSTM", output_path))
        extract_audio_latent_codes(model_name, t_latents, b_latents, size, output_path)

    torch.cuda.empty_cache()
    model = VQVAE2()
    try:
        t_path = join(root, "models/LSTM", output_path, t_latents)
        b_path = join(root, "models/LSTM", output_path, b_latents)
        top_latents = torch.load(t_path).to(device)
        bottom_latents = torch.load(b_path).to(device)
        
        audio_path = join(root, "models/LSTM", output_path, "audio.pt")
        audio_info = torch.load(audio_path).to(device)
    except Exception as e:
        print(f"Failed to load latents: {e} Exiting...")
        return -1
    
    try:
        model_path = join(root, "models", model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))   
    except Exception as e:
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["vqgan"])
        except Exception as e:
            print(f"Unable to load model: {e}. Exiting...")
    model.to(device)     

    if not exists(join(root, "models", "LSTM", output_path)):
        makedirs(join(root, "models", "LSTM", output_path))

    vocab_size = model.num_embeddings    
    top_sequence = top_latents.view(top_latents.size(0), -1)
    bottom_sequence = bottom_latents.view(bottom_latents.size(0), -1)
    
    t_dataset = AudioLatentDataset(top_sequence, audio_info)
    b_dataset = AudioInheritedLatentDataset(bottom_sequence, top_sequence, audio_info)   
    
    lstm = AudioLatentLSTM(vocab_size, model.num_embeddings, hidden_dim=4, layers=3, audio_dim=size, audio_embed_dim=512).to(device)
    bottom_lstm = AudioInheritedLatentLSTM(vocab_size, model.num_embeddings, hidden_dim=4, layers=3, audio_dim=size, audio_embed_dim=512).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(lstm.parameters(), lr=1e-3)
    
    t_dataloader = DataLoader(t_dataset, batch_size=32, shuffle=True)
    b_dataloader = DataLoader(b_dataset, batch_size=32, shuffle=True)
    
    if not load_top:
        
        last_saved = 0
        
        for epoch in range(num_epochs):
            for i, (inputs, target, audio) in enumerate(t_dataloader):
                print_progress_bar(epoch, i, len(t_dataloader))

                # Get items
                inputs, target, audio = inputs.to(device).long(), target.to(device).long(), audio.to(device)

                # Train LSTM
                optimiser.zero_grad()
                outputs, _ = lstm(inputs, audio)
                loss = criterion(outputs.view(-1, vocab_size), target.view(-1))
                loss.backward()
                optimiser.step()

            torch.cuda.empty_cache()
            
            if epoch % 25 == 0 and epoch > 0:        
                try:
                    torch.save(lstm.state_dict(), join(root, "models", "LSTM", output_path, f"BACKUP{epoch}_t_lstm.pth"))
                except:
                    print("Couldn't save top LSTM")
            
            if last_saved > 20:
                try:
                    torch.save(lstm.state_dict(), join(root, "models", "LSTM", output_path, f"t_lstm.pth"))
                    last_saved = 0
                except:
                    print("Couldn't save top LSTM")
            else:
                last_saved += 1
            
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
        if epoch % 25 == 0 and epoch > 0:        
            try:
                torch.save(lstm.state_dict(), join(root, "models", "LSTM", output_path, f"BACKUP{epoch}_b_lstm.pth"))
            except:
                print("Couldn't save top LSTM")
        
        if last_saved > 15:
            try:
                torch.save(lstm.state_dict(), join(root, "models", "LSTM", output_path, f"b_lstm.pth"))
                last_saved = 0
            except:
                print("Couldn't save top LSTM")
        else:
            last_saved += 1



# ALLOWS TRAINING OF VQVAE-2 and VQGAN
def train_vanilla_lstm_hierarchical(model_name, t_latents, b_latents, num_epochs=100, load_top=False):

    output_path = input("What is your desired output path/where are latents stored?: ")
    answer = input("Do you need to generate latent codes? (y/n): ")
    if answer == "y":
        if not exists(join(root, "models", "LSTM", output_path)):
            makedirs(join(root, "models", "LSTM", output_path))
        extract_latent_codes(model_name, t_latents, b_latents, size, output_path)

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
                torch.save(lstm.state_dict(), join(root, "models", "LSTM", output_path, "t_lstm.pth"))
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
            torch.save(bottom_lstm.state_dict(), join(root, "models", "LSTM", output_path, "b_lstm.pth"))
        except:
            print("\nFailed to save bottom LSTM")


def decode(image_model, stored_location, t_lstm, b_lstm, image_size, use_audio=False):
    
    torch.cuda.empty_cache()
    model = VQVAE2()
    try:
        
        if use_audio:
            top_lstm = AudioLatentLSTM(model.num_embeddings, model.num_embeddings, hidden_dim=3, layers=3, audio_dim=image_size, audio_embed_dim=512).to(device)
            bottom_lstm = AudioInheritedLatentLSTM(model.num_embeddings, model.num_embeddings, hidden_dim=3, layers=3, audio_dim=image_size, audio_embed_dim=512).to(device)
        else:
            top_lstm = LatentLSTM(model.num_embeddings, model.num_embeddings, hidden_dim=3, layers=3).to(device)
            bottom_lstm = InheritedLatentLSTM(model.num_embeddings, model.num_embeddings, hidden_dim=3, layers=3).to(device)

        top_lstm.load_state_dict(torch.load(join(root, "models/LSTM", stored_location, t_lstm), map_location=device))
        bottom_lstm.load_state_dict(torch.load(join(root, "models/LSTM", stored_location, b_lstm), map_location=device))
    
    except Exception as e:
        print(f"Failed to recover LSTMs: {e} Exiting...")
        return -1
    
    try:
        model.load_state_dict(torch.load(join(root, "models", image_model), map_location=device))
    except Exception as e:        
        try:
            checkpoint = torch.load(join(root, "models", image_model), map_location=device)
            model.load_state_dict(checkpoint["vqgan"])
        except Exception as e:
            print(f"Unable to load model: {e}. Exiting...")
            return    
    
    if not use_audio:
        top_latents, bottom_latents = sample_latents(top_lstm, bottom_lstm, 0, image_size * image_size)
    else:
        top_latents, bottom_latents = sample_latents_with_audio(top_lstm, bottom_lstm, 0, image_size, "VQGAN-FT128/t_latents.pt")
    
    
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
            top_generated.append(output.item())
            input_seq = output

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
    

def sample_latents_with_audio(lstm, bottom_lstm, start_token, size, t_latents, temperature=0.5):
    
    print("Sampling with audio")
    
    lstm.eval()
    bottom_lstm.eval()
    
    top_sequence_length = 32 * 32
    bottom_sequence_length = 64 * 64
    
    try:
        top_latents = torch.load(join(root, "models", "LSTM", t_latents)).to(device)
        top_sequence = top_latents.view(top_latents.size(0), -1)
        
        AudioDataset = CustomAudioImagePairing(join(root, "data/downloaded_images"), join(root, "data/spectrograms"), size)
        
    except Exception as e:
        print(f"Error: {e}, couldn't extract latents...")
        return -1, -1

    t_dataset = LatentDataset(top_sequence)    
    start_token, _ = t_dataset[0]
    start_token = start_token[0].item()

    audio_input, _ = AudioDataset[1]
    audio_input = audio_input.unsqueeze(0).to(device)
    
    top_generated = [start_token]
    input_seq = torch.tensor([[start_token]], device=device).long()

    for _ in range(top_sequence_length - 1):
        with torch.no_grad():
            output, _ = lstm(input_seq, audio_input)
            next_token = output[:, -1].argmax(dim=-1)
            top_generated.append(next_token.item())
            
            input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=-1).long()
            
    print("Sampled top latents")
            

    top_generated = torch.tensor(top_generated, device=device).unsqueeze(0).long()  # shape: (1, sequence_length)
    bottom_generated = []
    input_seq = torch.zeros((1, 1), device=device).long()  # Start with some neutral token

    for _ in range(bottom_sequence_length):  # 64 * 64 = 4096
        with torch.no_grad():
            output, _ = bottom_lstm(input_seq, top_generated, audio_input)
            probs = torch.softmax(output[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            bottom_generated.append(next_token.item())
            input_seq = next_token.unsqueeze(0)
    
    print("Sampled bottom latents")
    
    return top_generated, torch.tensor(bottom_generated, device=device)
        

if __name__ == "__main__":
    
    
    #decode("128x128/VQGAN-FT128.pth", "VQGAN-FT128", "t_lstm.pth", "b_lstm.pth", 128, True)
    
    
    while True:
        model_name = input("Please enter the name of your image model: ").strip()
        if model_name:
            if not model_name.endswith(".pth"):
                model_name += ".pth"
            break
        print("Model name cannot be empty.")

    while True:
        t_latents = input("What is the name of your top latents?: ").strip()
        if t_latents:
            if not t_latents.endswith(".pt"):
                t_latents += ".pt"
            break
        print("Top latents cannot be empty.")
        
    while True:
        b_latents = input("What is the name of your bottom latents?: ").strip()
        if b_latents:
            if not b_latents.endswith(".pt"):
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
                    train_audio_lstm_vae(model_name, t_latents, size, num_epochs)
                    break
                elif model_type == "2":
                    train_audio_lstm_hierarchical(model_name, t_latents, b_latents, size, num_epochs, Load)
                    break
                elif model_type == "3":
                    train_vanilla_lstm_hierarchical(model_name, t_latents, b_latents, num_epochs, Load)
                    break
                else:
                    print("Invalid input. Please enter 1, 2, or 3.")            
            break
        
        elif answer == "3":
            audio = input("Do you want to use audio? (y/n): ")
            use_audio = audio == "y"
            output_path = input("Where is your data stored? (LSTM/[?]): ")
            decode(model_name, output_path, "t_lstm.pth", "b_lstm.pth", size, use_audio)
            break
        
        else:
            print("Answer must be one of 1, 2, or 3.")