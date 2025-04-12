from vqvae import VQVAE
import torch
from torch import nn, optim
from torch.nn import functional
from utils import print_progress_bar, CustomAudioImagePairing
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = "C:/Users/chwah/Dropbox/Family/Christopher/University/Y3/Year Long Project/SynaesthesAI"

# NEED TO GET DATASET OF THE IMAGES AND THE MATCHING AUDIO. THEN WE ENCODE THE AUDIO AND GIVE IT THAT FUNKY LOSS FUNCTION

def train_audio_encoder(epochs=50):
            
    # create a VQ-VAE to encode the audio with, not using the decoder
    AudioEncoder = VQVAE(input_dim=3)
    ImageEncoder = VQVAE(input_dim=3)
    
    try:
        ImageEncoder.load_state_dict(torch.load(f"{base}/models/vae.pth", map_location=device))
        ImageEncoder.to(device)
        AudioEncoder.load_state_dict(torch.load(f"{base}/models/audioVAE.pth", map_location=device))
        AudioEncoder.to(device)
        
    except Exception as e:
        print(f"Failed to load existing VQ-VAE: {e}")
        return -1

    audio_dir = f"{base}/data/spectrograms"
    image_dir = f"{base}/data/downloaded_images"
    
    dataset = CustomAudioImagePairing(image_dir, audio_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    
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
            
            print("\nImage_t (target) std:", image_t.std().item())
            print("Audio_t (generated)std:", audio_t.std().item())
            
            print("\nImage_t (target) mean:", image_t.mean().item())
            print("Audio_t (generated) mean:", audio_t.mean().item())
            
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
    
    epochs = input("Please enter number of epochs: ")
    try:
        epochs = int(epochs)
    except:
        print("Invalid input - Defaulting to 10 Epochs")
        epochs = 10
    train_audio_encoder(epochs)