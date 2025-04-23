import os
import torch
from networks import VAE, VQVAE, VQVAE2
from PIL import Image
from utils import deconvolve, CustomImageFolder, CustomAudioImagePairing
import numpy as np
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "C:/Users/chwah/Dropbox/Family/Christopher/University/Y3/Year Long Project/SynaesthesAI"


def test_vqvae(input_model, model_type, image_size):
    
    model_name = input_model.split("/")[-1].split(".")[0]
    
    model_path = os.path.join(root, "models", input_model)
    if not os.path.exists(model_path):    
        print("Model not found.")
        return
    
    dataset = CustomImageFolder(f"{root}/data/test_images", image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    if model_type == VAE:
        model = model_type(model_image_size=image_size)
    else:
        model = model_type()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["vqgan"])
        except Exception as e:
            print(f"Unable to load model: {e}. Exiting...")
            traceback.print_exc()
            return
    
    
    model.to(device)
    
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)
        
        if model_type==VAE:
            recon_images, _, _ = model(images)
        else:    
            recon_images, _ = model(images)
            
        for i in range(len(recon_images)):        
            Image.fromarray((deconvolve(recon_images[i].cpu().detach().numpy().squeeze()).transpose(1,2,0) * 255).astype(np.uint8)).save(f"{root}/data/outputs/{model_name}.jpeg")



def test_audio_vqvae(root, audio_model, image_model):
    if not os.path.exists(audio_model) or not os.path.exists(image_model):    
        print("Model not found.")
        return
    dataset = CustomAudioImagePairing(f"{root}data/downloaded_images/", f"{root}data/spectrograms")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    image_vae = VQVAE2()
    image_vae.to(device)
    image_vae.load_state_dict(torch.load(image_model, map_location=device))
    
    audio_vae = VQVAE2()
    audio_vae.to(device)
    audio_vae.load_state_dict(torch.load(audio_model, map_location=device))
    
    
    for epoch in range(1):
        for i, (spectrograms, images) in enumerate(dataloader):
            
            audio_t, audio_b, _, _, _ = audio_vae.encode(spectrograms)
            
            #audio_t = (audio_encoded_t - -0.01742841675877571) / (0.13349851965904236 + 1e-5)
            #audio_t = audio_t * 0.24672505259513855 + -0.020536981523036957
            #
            #
            #audio_b = (audio_encoded_b - -0.01742841675877571) / (0.13349851965904236 + 1e-5)
            #audio_b = audio_b * 0.24672505259513855 + -0.020536981523036957
            
            
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
        
        model_files = []
        model_dir = os.path.join(root, "models")
        
        # Recursively find all .pth files in subdirectories
        for subdir, _, files in os.walk(model_dir):
            for file in files:
                if file.endswith(".pth"):
                    full_path = os.path.relpath(os.path.join(subdir, file), model_dir)
                    model_files.append(full_path)

        if not model_files:
            print("No .pth model files found in 'models/' directory.")
        else:
            print("Available VAE model files:")
            for f in model_files:
                print(f"  - {f}")

            while True:
                selected = input("Enter the name of your VAE model from the list above: ").strip()
                break

            while True:
                try:
                    size = int(input("Please enter the size of your images (max 256): ").strip())
                    if 0 < size <= 256:
                        break
                    else:
                        print("Size must be a positive number no greater than 256.")
                except ValueError:
                    print("Invalid input. Please enter a whole number.")


            while True:
                model = input("Are you testing a VAE [1], VQVAE [2], or VQVAE2 [3]?: ")
                if model == "1":
                    test_vqvae(selected, VAE, image_size=size)
                    break
                elif model == "2":
                    test_vqvae(selected, VQVAE, size)
                    break
                elif model == "3":
                    test_vqvae(selected, VQVAE2, size)
                    break
                else:
                    print("Invalid input. Please enter either 1, 2, or 3.")

    print("Done")
