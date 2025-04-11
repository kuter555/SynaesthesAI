from vqvae import VQVAE
import torch
from torch import nn, optim
from torch.nn import functional
from utils import print_progress_bar, CustomAudioImagePairing
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NEED TO GET DATASET OF THE IMAGES AND THE MATCHING AUDIO. THEN WE ENCODE THE AUDIO AND GIVE IT THAT FUNKY LOSS FUNCTION

def train_audio_encoder(epochs=50):
            
    # create a VQ-VAE to encode the audio with, not using the decoder
    AudioEncoder = VQVAE(input_dim=3)
    AudioEncoder.to(device)
    
    ImageEncoder = VQVAE(input_dim=3)
    try:
        ImageEncoder.load_state_dict(torch.load("../models/vae.pth", map_location=device))
        ImageEncoder.to(device)
    except Exception as e:
        print(f"Failed to load existing VQ-VAE: {e}")
        return -1
    
    base_dir = Path(__file__).resolve().parent

    # Safely build paths
    audio_dir = base_dir.parent / "data" / "spectrograms"
    image_dir = base_dir.parent / "data" / "downloaded_images"
    
    dataset = CustomAudioImagePairing(image_dir, audio_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    
    optimiser = optim.Adam(AudioEncoder.parameters(), lr=1e-3)
    mse_loss = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        for i, (spectrograms, images) in enumerate(dataloader):
            
            print_progress_bar(epoch, i, len(dataloader))
            
            # convert images for pytorch
            images = images.to(device)
            spectrograms = spectrograms.to(device)            
            image_t, image_b, _, _, _ = ImageEncoder.encode(images)
            
            # actual training    
            optimiser.zero_grad()
            audio_t, audio_b, _, _, _ = AudioEncoder.encode(spectrograms)
            
            # LOSS FUNCTION
            vector_loss_t = mse_loss(audio_t, image_t)
            vector_loss_b = mse_loss(audio_b, image_b)
            loss = vector_loss_b + vector_loss_t            
            
            # STEP OPTIMISER
            loss.backward()
            optimiser.step()
            
        try:
            torch.save(AudioEncoder.state_dict(), "../models/audioVAE.pth")
        except:
            print("\nFailed to save vae")
        
        torch.cuda.empty_cache()
            
            
if __name__ == "__main__":
    
    train_audio_encoder()