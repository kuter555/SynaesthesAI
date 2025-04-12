import os
import torch
from vqvae import VQVAE
from PIL import Image
from utils import deconvolve, CustomImageFolder, CustomAudioImagePairing
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = "../models/vae.pth"

def test_vqvae(model):
    
    if not os.path.exists(model):    
        print("Model not found.")
        return
    
    dataset = CustomImageFolder("../data/test_images")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

    model = VQVAE()
    model.to(device)
    model.load_state_dict(torch.load(model, weights_only=True, map_location=device))
    

    for i, (images, _) in enumerate(dataloader):
        
        images = images.to(device)
        recon_images, _ = model(images)
        for i in range(len(recon_images)):        
            Image.fromarray((deconvolve(recon_images[i].cpu().detach().numpy().squeeze()).transpose(1,2,0) * 255).astype(np.uint8)).save(f".test_images/recon_{i}.jpeg")


def test_audio_vqvae(root, audio_model, image_model):
    if not os.path.exists(audio_model) or not os.path.exists(image_model):    
        print("Model not found.")
        return
    dataset = CustomAudioImagePairing(f"{root}data/downloaded_images/", f"{root}data/spectrograms")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    image_vae = VQVAE()
    image_vae.to(device)
    image_vae.load_state_dict(torch.load(image_model, map_location=device))
    
    audio_vae = VQVAE()
    audio_vae.to(device)
    audio_vae.load_state_dict(torch.load(audio_model, map_location=device))
    
    
    for epoch in range(1):
        for i, (spectrograms, images) in enumerate(dataloader):
            
            audio_encoded_t, audio_encoded_b, _, _, _ = audio_vae.encode(spectrograms)
            
            audio_t = (audio_encoded_t - -0.01742841675877571) / (0.13349851965904236 + 1e-5)
            audio_t = audio_t * 0.24672505259513855 + -0.020536981523036957
            
            
            audio_b = (audio_encoded_b - -0.01742841675877571) / (0.13349851965904236 + 1e-5)
            audio_b = audio_b * 0.24672505259513855 + -0.020536981523036957
            
            
            image_output = image_vae.decode(audio_t, audio_b)
    
            Image.fromarray((deconvolve(image_output[0].cpu().detach().numpy().squeeze()).transpose(1,2,0) * 255).astype(np.uint8)).save(f"{root}data/outputs/first_audio_test{i}.jpeg")

    
    
root = "C:/Users/chwah/Dropbox/Family/Christopher/University/Y3/Year Long Project/SynaesthesAI/"
audio_model = f"{root}models/audioVAE.pth"
image_model = f"{root}models/vae.pth"


if __name__ == "__main__":
    
    print("Testing novel image generation...")
    test_audio_vqvae(root, audio_model, image_model)
    print("Done")