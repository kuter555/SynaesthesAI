from networks import VAE, VQVAE2
import torch
from torch import optim
from utils import print_progress_bar, CustomAudioImagePairing

from dotenv import load_dotenv
from os import getenv, makedirs 
from os.path import join, exists

load_dotenv()
root = getenv('root')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Works for VQVAE-2 and VQGAN
def train_audio_encoder(img_model_name, audio_model_name, image_size, model_type, epochs=500, load=True):
            
    # create a VQ-VAE to encode the audio with, not using the decoder
    AudioEncoder = model_type(input_dim=3)
    ImageEncoder = model_type(input_dim=3)
    
    try:
        img_model_path = join(root, "models", img_model_name)
        ImageEncoder.load_state_dict(torch.load(img_model_path, map_location=device))
    except:
        try:
            checkpoint = torch.load(img_model_path, map_location=device)
            ImageEncoder.load_state_dict(checkpoint["vqgan"])
        except Exception as e:
            print(f"Unable to load model: {e}. Exiting...")
            return -1
    
    if load:            
        try:
            audio_model_path = join(root, "models", audio_model_name)
            AudioEncoder.load_state_dict(torch.load(audio_model_path, map_location=device))
        except:
            try:
                checkpoint = torch.load(audio_model_path, map_location=device)
                AudioEncoder.load_state_dict(checkpoint["vqgan"])
            except Exception as e:
                print(f"Unable to load model: {e}. Exiting...")
                return -1
            
    print("Successfully loaded models")
    
    ImageEncoder.to(device)
    AudioEncoder.to(device)
        
    image_dir = join(root, "data/downloaded_images")
    audio_dir = join(root, "data/spectrograms")
    
    dataset = CustomAudioImagePairing(image_dir, audio_dir, size=image_size)
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
            
            # LOSS FUNCTION
            vector_loss_t = mse_loss(audio_t, image_t)
            vector_loss_b = mse_loss(audio_b, image_b)
            loss = vector_loss_b + vector_loss_t            
            
            # STEP OPTIMISER
            loss.backward()
            optimiser.step()
            
        try:
            torch.save(AudioEncoder.state_dict(), join(root, "models", audio_model_name))
        except:
            print("\nFailed to save vae")
        
        torch.cuda.empty_cache()
            
            
            
# Works for VQVAE-2 and VQGAN
def train_audio_encoder_vae(img_model_name, audio_model_name, image_size, epochs=500, load=True, beta=0.25):
            
    AudioEncoder = VAE(input_dim=3)
    ImageEncoder = VAE(input_dim=3)
    
    try:
        img_model_path = join(root, "models", img_model_name)
        ImageEncoder.load_state_dict(torch.load(img_model_path, map_location=device))
    except Exception as e:
        print(f"Unable to load model: {e}. Exiting...")
        return -1
    
    if load:
        
        try:
            audio_model_path = join(root, "models", audio_model_name)
            AudioEncoder.load_state_dict(torch.load(audio_model_path, map_location=device))
        except Exception as e:
            print(f"Unable to load model: {e}. Exiting...")
            return -1
    
    ImageEncoder.to(device)
    AudioEncoder.to(device)
        
    image_dir = join(root, "data/downloaded_images")
    audio_dir = join(root, "data/spectrograms")
    
    dataset = CustomAudioImagePairing(image_dir, audio_dir, image_size=image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    
    optimiser = optim.Adam(AudioEncoder.parameters(), lr=1e-3)
    mse_loss = torch.nn.MSELoss()
    
    
    # KL-divergence to improve the loss of variance
    def kl_divergence_gaussians(mu1, logvar1, mu2, logvar2):
        return 0.5 * torch.sum(
            logvar2 - logvar1 +
            (torch.exp(logvar1) + (mu1 - mu2)**2) / torch.exp(logvar2) - 1
        )
    
    for epoch in range(epochs):
        for i, (spectrograms, images) in enumerate(dataloader):
            
            print_progress_bar(epoch, i, len(dataloader))
            
            # convert images for pytorch
            images = images.to(device)
            spectrograms = spectrograms.to(device)            
            img_mean, img_var = ImageEncoder.encode(images)
            
            # actual training    
            optimiser.zero_grad()
            audio_mean, audio_var = AudioEncoder.encode(spectrograms)
            
            # LOSS FUNCTION
            mean_loss = mse_loss(audio_mean, img_mean)
            var_loss = kl_divergence_gaussians(audio_mean, audio_var, img_mean, img_var)
            loss = mean_loss + beta * var_loss            
            
            # STEP OPTIMISER
            loss.backward()
            optimiser.step()
            
        try:
            torch.save(AudioEncoder.state_dict(), join(root, "models", audio_model_name))
        except:
            print("\nFailed to save vae")
        
        torch.cuda.empty_cache()
            
            
            
            
            
            
if __name__ == "__main__":
    
    
    
    while True:
        image_model = input("Please enter the name of your image model: ").strip()
        if image_model:
            if not image_model.endswith(".pth"):
                image_model += ".pth"
            break
        print("Model name cannot be empty.")
        
        
    while True:
        audio_model = input("Please enter the name of your audio model: ").strip()
        if audio_model:
            if not audio_model.endswith(".pth"):
                audio_model += ".pth"
            break
        print("Model name cannot be empty.")    
    
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
        answer = input("Would you like to load existing audio models? (y/n): ")
        if answer.strip().lower() in ["y", "n"]:
            load = answer == "y"
            break
        print("Invalid input. Please enter either y or n.")
    
      
    while True:
        model_type = input("What model? VAE [1], or VQVAE2/VQGAN-FT [2]?: ").strip()
        if model_type == "1":
            train_audio_encoder_vae(image_model, audio_model, size, num_epochs, load=load, beta=0.25)
            break
        elif model_type == "2":
            train_audio_encoder(image_model, audio_model, size, VQVAE2, num_epochs, load=load)
            break
        else:
            print("Invalid input. Please enter 1 or 2.")
