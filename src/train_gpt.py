import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from train_vqvae import train_vqvae
from utils import print_progress_bar, HierarchicalLatentDataset, LatentDataset, extract_latent_codes, extract_audio_latent_codes_gpt, AudioHierarchicalLatentDataset, AudioLatentDataset
from networks import VAE, VQVAE, VQVAE2, LatentGPT, BottomGPT

import os
from dotenv import load_dotenv

load_dotenv()
root = os.getenv('root')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 1e-3


# ALLOWS TRAINING OF VQVAE-2 and VQGAN
def train_audio_gpt_hierarchical(model_name, t_latents, b_latents, size, num_epochs=100, load_top=False):


    # Load/Create Audio Model
    audio_filename = input("What is your audio model name?: ")
    answer = input("Do you need to train an audio encoder? (y/n): ")
    if answer.strip().lower() == "y":
        train_vqvae(audio_filename, VQVAE, data_file="spectrograms", epochs=250)
    
    audio_model = VQVAE()
    try:
        audio_path = os.path.join(root, "models", audio_filename)
        audio_model.load_state_dict(torch.load(audio_path, map_location=device))
    except Exception as e:
        print(f"Failed loading audio vqvae: {e}. Exiting...")
        return -1
    audio_model.to(device)

    # Generate Latents
    output_path = input("What is your desired output path/where are latents stored?: ")
    answer = input("Do you need to generate latent codes? (y/n): ")
    if answer == "y":
        if not os.path.exists(os.path.join(root, "models", "GPT", output_path)):
            os.makedirs(os.path.join(root, "models", "GPT", output_path))
        extract_audio_latent_codes_gpt(model_name, audio_filename, t_latents, b_latents, size, output_path)

    torch.cuda.empty_cache()
    model = VQVAE2()
    try: 
        t_path = os.path.join(root, "models", t_latents)
        b_path = os.path.join(root, "models", b_latents)
        top_latents = torch.load(t_path).to(device)
        bottom_latents = torch.load(b_path).to(device)
        
        audio_path = os.path.join(root, "models/GPT", output_path, "audio.pt")
        audio_info = torch.load(audio_path).to(device)

    except Exception as e:
        print(f"Failed to run: {e} Exiting...")
        return -1
    
    try:
        model_path = os.path.join(root, "models", model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))   
    except Exception as e:
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["vqgan"])
        except Exception as e:
            print(f"Unable to load model: {e}. Exiting...")
    model.to(device)
    
    if not os.path.exists(os.path.join(root, "models", "GPT", output_path)):
        os.makedirs(os.path.join(root, "models", "GPT", output_path))

    vocab_size = model.num_embeddings
    top_sequence = top_latents.view(top_latents.size(0), -1)
    bottom_sequence = bottom_latents.view(bottom_latents.size(0), -1)
    
    t_dataset = AudioLatentDataset(top_sequence, audio_info)
    b_dataset = AudioHierarchicalLatentDataset(top_sequence, bottom_sequence, audio_info)   
    
    t_dataloader = DataLoader(t_dataset, batch_size=32, shuffle=True)
    b_dataloader = DataLoader(b_dataset, batch_size=32, shuffle=True)
    
    t_model = LatentGPT(vocab_size=model.num_embeddings).to(device)
    b_model = BottomGPT(vocab_size=model.num_embeddings).to(device)
    t_optimiser = torch.optim.Adam(t_model.parameters(), lr=lr)
    b_optimiser = torch.optim.Adam(b_model.parameters(), lr=lr)  
        
    if not load_top:
        print("Training top GPT...")
        for epoch in range(num_epochs):
            for i, (inputs, target) in enumerate(t_dataloader):
                print_progress_bar(epoch, i, len(t_dataloader))

                # Get items
                inputs, target = inputs.to(device).long(), target.to(device).long()

                # Train LSTM
                logits = t_model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target.view(-1)
                )
                                
                loss.backward()
                t_optimiser.step()
                t_optimiser.zero_grad()
                
            try:
                torch.save(t_model.state_dict(), os.path.join(root, "models", "GPT", output_path, "t_gpt.pth"))
            except:
                print("Couldn't save top GPT")
                
            if epoch % 50 == 0:
                try:
                    torch.save(t_model.state_dict(), os.path.join(root, "models", "GPT", output_path, f"BACKUP{epoch}-t_gpt.pth"))
                except Exception as e:
                    print(f"Couldn't save top GPT backup: {e}")    
            

            torch.cuda.empty_cache()
            
    # train the bottom LSTM
    for epoch in range(num_epochs):
        for i, (input, target) in enumerate(b_dataloader):
            print_progress_bar(epoch, i, len(b_dataloader))
            
            input, target = input.to(device), target.to(device)
            logits = b_model(input) 
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=-100
            )
                   
            loss.backward()
            b_optimiser.step()
            b_optimiser.zero_grad()
            
        try:
            torch.save(b_model.state_dict(), os.path.join(root, "models", "GPT", output_path, "b_gpt.pth"))
        except:
            print("Couldn't save bottom GPT")
            
        if epoch % 50 == 0:
            try:
                torch.save(b_model.state_dict(), os.path.join(root, "models", "GPT", output_path, f"BACKUP{epoch}-b_gpt.pth"))
            except Exception as e:
                print(f"Couldn't save bottom GPT backup: {e}")    
        
        torch.cuda.empty_cache()







# ALLOWS TRAINING OF VQVAE-2 and VQGAN
def train_vanilla_gpt_hierarchical(model_name, t_latents, b_latents, size, num_epochs=100, load_top=False):

    output_path = input("What is your desired output path/where are latents stored?: ")
    answer = input("Do you need to generate latent codes? (y/n): ")
    if answer == "y":
        if not os.path.exists(os.path.join(root, "models", "GPT", output_path)):
            os.makedirs(os.path.join(root, "models", "GPT", output_path))
        extract_latent_codes(model_name, t_latents, b_latents, size, output_path)


    torch.cuda.empty_cache()
    model = VQVAE2()
    try:
        t_path = os.path.join(root, "models", t_latents)
        b_path = os.path.join(root, "models", b_latents)
        top_latents = torch.load(t_path).to(device)
        bottom_latents = torch.load(b_path).to(device)    

    except Exception as e:
        print(f"Failed to run: {e} Exiting...")
        return -1        
    
    try:
        model_path = os.path.join(root, "models", model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))   
    except Exception as e:
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["vqgan"])
        except Exception as e:
            print(f"Unable to load model: {e}. Exiting...")
    model.to(device)
    
    
    if not os.path.exists(os.path.join(root, "models", "GPT", output_path)):
        os.makedirs(os.path.join(root, "models", "GPT", output_path))

    vocab_size = model.num_embeddings
    top_sequence = top_latents.view(top_latents.size(0), -1)
    bottom_sequence = bottom_latents.view(bottom_latents.size(0), -1)
    
    t_dataset = LatentDataset(top_sequence)
    b_dataset = HierarchicalLatentDataset(top_sequence, bottom_sequence)   
    
    t_dataloader = DataLoader(t_dataset, batch_size=32, shuffle=True)
    b_dataloader = DataLoader(b_dataset, batch_size=32, shuffle=True)
    
    t_model = LatentGPT(vocab_size=model.num_embeddings).to(device)
    b_model = BottomGPT(vocab_size=model.num_embeddings).to(device)
    t_optimiser = torch.optim.Adam(t_model.parameters(), lr=lr)
    b_optimiser = torch.optim.Adam(b_model.parameters(), lr=lr)  
        
    if not load_top:
        print("Training top GPT...")
        for epoch in range(num_epochs):
            for i, (inputs, target) in enumerate(t_dataloader):
                print_progress_bar(epoch, i, len(t_dataloader))

                # Get items
                inputs, target = inputs.to(device).long(), target.to(device).long()

                # Train LSTM
                logits = t_model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target.view(-1)
                )
                                
                loss.backward()
                t_optimiser.step()
                t_optimiser.zero_grad()
                
            try:
                torch.save(t_model.state_dict(), os.path.join(root, "models", "GPT", output_path, "t_gpt.pth"))
            except:
                print("Couldn't save top GPT")
                
            if epoch % 50 == 0:
                try:
                    torch.save(t_model.state_dict(), os.path.join(root, "models", "GPT", output_path, f"BACKUP{epoch}-t_gpt.pth"))
                except Exception as e:
                    print(f"Couldn't save top GPT backup: {e}")    
            

            torch.cuda.empty_cache()
            
    # train the bottom LSTM
    for epoch in range(num_epochs):
        for i, (input, target) in enumerate(b_dataloader):
            print_progress_bar(epoch, i, len(b_dataloader))
            
            input, target = input.to(device), target.to(device)
            logits = b_model(input) 
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=-100
            )
                   
            loss.backward()
            b_optimiser.step()
            b_optimiser.zero_grad()
            
        try:
            torch.save(b_model.state_dict(), os.path.join(root, "models", "GPT", output_path, "b_gpt.pth"))
        except:
            print("Couldn't save bottom GPT")
            
        if epoch % 50 == 0:
            try:
                torch.save(b_model.state_dict(), os.path.join(root, "models", "GPT", output_path, f"BACKUP{epoch}-b_gpt.pth"))
            except Exception as e:
                print(f"Couldn't save bottom GPT backup: {e}")    
        
        torch.cuda.empty_cache()











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
        answer = input("Would you like to Extract Latents (1) or Train the GPT (2), or Decode (3)? > ")
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
                    #train_audio_lstm_vae(model_name, t_latents, size, num_epochs)
                    break
                elif model_type == "2":
                    train_audio_gpt_hierarchical(model_name, t_latents, b_latents, size, num_epochs, Load)
                    break
                elif model_type == "3":
                    train_vanilla_gpt_hierarchical(model_name, t_latents, b_latents, num_epochs, Load)
                    break
                else:
                    print("Invalid input. Please enter 1, 2, or 3.")            
            break
        
        elif answer == "3":
            break
        
        else:
            print("Answer must be one of 1, 2, or 3.")