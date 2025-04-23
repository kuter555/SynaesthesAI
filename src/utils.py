import torch
from torch import from_numpy
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F

from PIL import Image
import numpy as np

import os
import sys
from dotenv import load_dotenv

from networks import VAE, VQVAE, VQVAE2

load_dotenv()
root = os.getenv('root')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HierarchicalLatentDataset(Dataset):
    def __init__(self, top_seqs, bottom_seqs):
        self.top_seqs = top_seqs  # shape: (N, 1024)
        self.bottom_seqs = bottom_seqs  # shape: (N, 4096)

    def __len__(self):
        return len(self.top_seqs)

    def __getitem__(self, idx):
        top = self.top_seqs[idx]
        bottom = self.bottom_seqs[idx]

        input_ids = torch.cat([top, bottom], dim=0)
        target_ids = torch.cat([
            torch.full_like(top, -100),  # mask top tokens from loss
            bottom
        ], dim=0)

        return input_ids, target_ids


class AudioHierarchicalLatentDataset(Dataset):
    def __init__(self, top_seqs, bottom_seqs, audio):
        self.top_seqs = top_seqs  # shape: (N, 1024)
        self.bottom_seqs = bottom_seqs  # shape: (N, 4096)
        self.audio = audio

    def __len__(self):
        return len(self.top_seqs)

    def __getitem__(self, idx):
        top = self.top_seqs[idx]
        bottom = self.bottom_seqs[idx]

        input_ids = torch.cat([top, bottom], dim=0)
        target_ids = torch.cat([
            torch.full_like(top, -100),  # mask top tokens from loss
            bottom
        ], dim=0)
        audio = self.audio[idx]
        
        return input_ids, target_ids, audio






class AudioLatentDataset(Dataset):
    def __init__(self, sequences, audio):
        self.data = sequences
        self.audio = audio
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        aud = self.audio[idx]
        return sequence[:-1], sequence[1:], aud


# need this one conditioned off of the top dataset
class AudioInheritedLatentDataset(Dataset):
    def __init__(self, b_sequences, sequences, audio):
        self.b_sequences = b_sequences
        self.sequences = sequences
        self.audio = audio
    
    def __len__(self):
        return len(self.b_sequences)
    
    def __getitem__(self, idx):
        b_sequence = self.b_sequences[idx]
        sequence = self.sequences[idx]
        aud = self.audio[idx]
        return sequence, b_sequence[:-1], b_sequence[1:], aud  




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
    

class CustomAudioImagePairing(Dataset):
    
    def __init__(self, image_dir, audio_dir, size):
        self.image_dir = image_dir
        self.audio_dir = audio_dir
        
        self.images = []
        self.audio = []
        
        self.size = size
        
        self.transform =  transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5, 0.5), (0.5,0.5,0.5))
        ])       
        
         # Collect all valid image files
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith('.jpeg'):
                    try:
                        path = os.path.join(root, file)
                        with Image.open(path) as img:
                            img.verify()
                        self.images.append(file.split('.')[0])
                    except:
                        continue
                    
                    
        for _, _, x_files in os.walk(self.audio_dir):
            for file in x_files:
                if file.lower().endswith('.npy'):
                    try:
                        path = file.split('.')[0]
                        self.audio.append(path)
                    except:
                        continue
        
        
        self.pairings = []
        for song in self.audio:
            if song in self.images:
                self.pairings.append(song)                
        

        
    def __len__(self):
        return len(self.pairings)
    
    
    def __getitem__(self, idx):
        pair = self.pairings[idx]
        image = Image.open(os.path.join(f"{self.image_dir}", f"{pair}.jpeg")).convert("RGB")  # Open image and ensure it's RGB
        if self.transform:
            image = self.transform(image)
        
        spectrogram = np.load(os.path.join(self.audio_dir, f"{pair}.npy"))
        spectrogram = from_numpy(spectrogram).float()  # convert to tensor
        
        if spectrogram.ndim == 3:
            spectrogram = spectrogram.unsqueeze(0)
        elif spectrogram.ndim == 2:
            spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)
            
        spectrogram = F.interpolate(spectrogram, size=(self.size, self.size), mode='bilinear', align_corners=False)
        spectrogram = spectrogram.repeat(1, 3, 1, 1)
        spectrogram = spectrogram.squeeze(0)
        
        return spectrogram, image       
    


# CHATGPT ASSISTED
class CustomImageFolder(Dataset):
    def __init__(self, root_dir, image_size):
        self.root_dir = root_dir
        self.image_files = []
        
        self.transform =  transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5, 0.5), (0.5,0.5,0.5))
        ])
        
        # Supported image extensions
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        # Collect all valid image files
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.valid_extensions):
                    try:
                        full_path = os.path.join(root, file)
                        
                        with Image.open(full_path) as img:
                            img.verify()  # Verify image integrity
                        self.image_files.append(full_path)
                        
                    except:
                        continue
     
    
    def __len__(self):
        return len(self.image_files)
    
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        try:
            image = Image.open(image_path).convert("RGB")  # Open image and ensure it's RGB
        
            if self.transform:
                image = self.transform(image)
            return image, 0  # We don't need labels for VAE (return dummy label 0)
        
        except:
            return self.__getitem__((idx + 1) % len(self))


def print_progress_bar(epoch, iteration, total, length=50):
    progress = int(length * iteration / total)
    if epoch != -1:
        bar = f"\033[31m Epoch {epoch}:\033[97m [{'=' * progress}{' ' * (length - progress)}] {int(100 * iteration / total)}%"
    else:
        bar = f"[{'=' * progress}{' ' * (length - progress)}] {int(100 * iteration / total)}%"
    sys.stdout.write(f"\r{bar}")
    sys.stdout.flush()
    
    
    
    
    
def extract_audio_latent_codes_vae(model_path, latent_name, image_size, output_path):
    
    print("Beginning Extraction")
    
    dataset = CustomAudioImagePairing(f"{root}/data/downloaded_images", audio_dir=f"{root}/data/spectrograms", image_size=image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    
    model = VAE()
    model_path = os.path.join(root, "models", model_path)
    print("Loading Model Dict")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Unable to load model: {e}. Exiting...")
    model.to(device)
    
    print("Starting image processing")
    
    with torch.no_grad():
        model.eval()
        stored_latents = []
        stored_audio = []
        for i, (audio, images) in enumerate(dataloader):
            print_progress_bar("Extracting", i, len(dataloader))
            
            images = images.to(device)
            
            mean, var = model.encode(images)
            z = model.reparameterization(mean, var)
            
            stored_latents.append(z.cpu())
            stored_audio.append(audio.cpu())
    
    stored_latents = torch.cat(stored_latents, dim=0)
    stored_audio = torch.cat(stored_audio, dim=0)
        
    head_path = os.path.join(root, "models", "LSTM", output_path)
        
    torch.save(stored_latents, os.path.join(head_path, latent_name))
    torch.save(stored_audio, os.path.join(head_path, "audio.pt"))
            
    print("Latents successfully saved!")   
    
    
    

    
def extract_audio_latent_codes_gpt(model_path, audio_model_path, t_latent_name, b_latent_name, image_size, output_path):
        
    print("Beginning Extraction")
    
    dataset = CustomAudioImagePairing(f"{root}/data/downloaded_images", audio_dir=f"{root}/data/spectrograms", image_size=image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    
    model = VQVAE2()
    model_path = os.path.join(root, "models", model_path)
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
    
    audio_model = VQVAE()
    audio_model_path = os.path.join(root, "models", audio_model_path)
    print("Loading Model Dict")
    try:
        audio_model.load_state_dict(torch.load(audio_model_path, map_location=device))
    except:
        try:
            checkpoint = torch.load(audio_model_path, map_location=device)
            audio_model.load_state_dict(checkpoint["vqgan"])
        except Exception as e:
            print(f"Unable to load audio model: {e}. Exiting...")
    audio_model.to(device)
    
    print("Starting image processing")
    
    with torch.no_grad():
        model.eval()
        stored_latent_t = []
        stored_latent_b = []
        stored_audio = []
        for i, (audio, images) in enumerate(dataloader):
            
            print_progress_bar("Extracting", i, len(dataloader))
            
            images = images.to(device)
            _, _, _, index_t, index_b = model.encode(images)
            _, indices, _ = audio_model.encode(audio)

            stored_latent_t.append(index_t.cpu())
            stored_latent_b.append(index_b.cpu())
            stored_audio.append(indices.cpu())
    
    stored_latent_b = torch.cat(stored_latent_b, dim=0)
    stored_latent_t = torch.cat(stored_latent_t, dim=0)
    stored_audio = torch.cat(stored_audio, dim=0)
        
    head_path = os.path.join(root, "models", "LSTM", output_path)
        
    torch.save(stored_latent_t, os.path.join(head_path, t_latent_name))
    torch.save(stored_latent_b, os.path.join(head_path, b_latent_name))
    torch.save(stored_audio, os.path.join(head_path, "audio.pt"))
            
    print("Latents successfully saved!")
    
    
    
    
    
def extract_audio_latent_codes(model_path, t_latent_name, b_latent_name, image_size, output_path):
        
    print("Beginning Extraction")
    
    dataset = CustomAudioImagePairing(f"{root}/data/downloaded_images", audio_dir=f"{root}/data/spectrograms", image_size=image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    
    model = VQVAE2()
    model_path = os.path.join(root, "models", model_path)
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
        stored_audio = []
        for i, (audio, images) in enumerate(dataloader):
            
            print_progress_bar("Extracting", i, len(dataloader))
            
            images = images.to(device)
            _, _, _, index_t, index_b = model.encode(images)

            stored_latent_t.append(index_t.cpu())
            stored_latent_b.append(index_b.cpu())
            stored_audio.append(audio.cpu())
    
    stored_latent_b = torch.cat(stored_latent_b, dim=0)
    stored_latent_t = torch.cat(stored_latent_t, dim=0)
    stored_audio = torch.cat(stored_audio, dim=0)
        
    head_path = os.path.join(root, "models", "LSTM", output_path)
        
    torch.save(stored_latent_t, os.path.join(head_path, t_latent_name))
    torch.save(stored_latent_b, os.path.join(head_path, b_latent_name))
    torch.save(stored_audio, os.path.join(head_path, "audio.pt"))
            
    print("Latents successfully saved!")
    

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


def deconvolve(x):
    return ((x * 0.5) + 0.5).clip(0, 1)