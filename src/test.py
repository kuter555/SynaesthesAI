import os
import torch
from vqvae import VQVAE
from PIL import Image
from utils import deconvolve, CustomImageFolder, CustomAudioImagePairing
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "C:/Users/chwah/Dropbox/Family/Christopher/University/Y3/Year Long Project/SynaesthesAI"

def test_vqvae(input_model, image_size):
    
    model_path = os.path.join(root, "models", input_model)
    
    if not os.path.exists(model_path):    
        print("Model not found.")
        return
    
    dataset = CustomImageFolder(f"{root}/data/test_images", image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    model = VQVAE()
    model.to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["vqgan"])
        except Exception as e:
            print(f"Unable to load model: {e}. Exiting...")
        
        
    for i, (images, _) in enumerate(dataloader):
        
        images = images.to(device)
        recon_images, _ = model(images)
        for i in range(len(recon_images)):        
            Image.fromarray((deconvolve(recon_images[i].cpu().detach().numpy().squeeze()).transpose(1,2,0) * 255).astype(np.uint8)).save(f"{root}/data/outputs/recon_{input_model}.jpeg")



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

    
if __name__ == "__main__":
    
    audio_model = "models/audioVAE.pth"
        
    while True:
        answer = input("Do you want to test image [1] or audio [2]?: ").strip()
        if answer in ["1", "2"]:
            break
        print("Invalid input. Please enter 1 or 2.")

    if answer == "2":
        test_audio_vqvae(root, "/models/audioVAE.pth", "/models/vae.pth")
    else:
        model_dir = os.path.join(root, "models")
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]

        if not model_files:
            print("No .pth model files found in 'models/' directory.")
        else:
            print("Available VAE model files:")
            for f in model_files:
                print(f"  - {f}")

            while True:
                selected = input("Enter the name of your VAE model from the list above: ").strip()
                if selected in model_files:
                    break
                print("Invalid selection. Please choose from the list.")

            while True:
                try:
                    size = int(input("Please enter the size of your images (max 256): ").strip())
                    if 0 < size <= 256:
                        break
                    else:
                        print("Size must be a positive number no greater than 256.")
                except ValueError:
                    print("Invalid input. Please enter a whole number.")

            test_vqvae(selected, image_size=size)

    print("Done")
