import sys
from torchvision import transforms
import torch
from torch import from_numpy
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np


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
    
    def __init__(self, image_dir, audio_dir):
        self.image_dir = image_dir
        self.audio_dir = audio_dir
        
        self.images = []
        self.audio = []
        
        self.transform =  transforms.Compose([
            transforms.Resize((256, 256)),
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
            
        spectrogram = F.interpolate(spectrogram, size=(256, 256), mode='bilinear', align_corners=False)
        spectrogram = spectrogram.repeat(1, 3, 1, 1)
        spectrogram = spectrogram.squeeze(0)
        
        return spectrogram, image       
    


# CHATGPT ASSISTED
class CustomImageFolder(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = []
        
        self.transform =  transforms.Compose([
            transforms.Resize((256, 256)),
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
    
    

def deconvolve(x):
    return ((x * 0.5) + 0.5).clip(0, 1)