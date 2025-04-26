import torch
from torch import optim
import torch.nn.functional as F

from networks import VAE, VQVAE, VQVAE2
from utils import print_progress_bar, CustomImageFolder, CustomAudioFolder, deconvolve

from os import getenv
from os.path import join
from dotenv import load_dotenv

load_dotenv()
root = getenv('root')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 1e-3
beta = 0.5

def train_vae(model_name, epochs=500, load=False, image_size=256):
    
    # VAE Loss Function
    def loss_function(x, x_hat, mean, log_var):        
        reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + KLD
    
    # Load data
    dataset = CustomImageFolder(f"{root}/data/downloaded_images", image_size=image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    # Create and load model
    model_path = join(root, "models", model_name)
    model = VAE(model_image_size=image_size)
    if(load):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Establish optimiser
    optimiser = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for i, (images, _) in enumerate(dataloader):
            print_progress_bar(epoch, i, len(dataloader))
            
            # Convert images for pytorch
            images = images.to(device)
            
            # Actual training
            optimiser.zero_grad()
            recon_images, mean, var = model(images)
            recon_loss = loss_function(deconvolve(images), deconvolve(recon_images), mean, var)
            recon_loss.backward()
            optimiser.step()
            
        # Save model
        try:
            torch.save(model.state_dict(), model_path)
        except:
            print(f"\nFailed to save at epoch: {epoch}")
        
        # Infrequent backups
        if epoch > 0 and epoch % 25 == 0:
            try:
                torch.save(model.state_dict(), join(root, "models", f"BACKUP{epoch}-{model_name}"))
            except:
                print(f"\nFailed to save backup at: {epoch}")
        torch.cuda.empty_cache()
    

    
def train_vqvae(model_name, model_type, data_file="downloaded_images", epochs=500, load=False, image_size=256):
    
    # Load data for training
    dataset = CustomImageFolder(f"{root}/data/{data_file}", image_size=image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    # Create and load model
    model_path = join(root, "models", model_name)
    model = model_type()
    if(load):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Establish optimisers
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = torch.nn.MSELoss()
    
    last_saved = 0
    
    # Training
    for epoch in range(epochs):
        for i, (images, _) in enumerate(dataloader):
            print_progress_bar(epoch, i, len(dataloader))
            
            # Convert images for pytorch
            images = images.to(device)
        
            # Actual training    
            optimiser.zero_grad()
            recon_images, quanitsed_loss = model(images)    
            recon_loss = mse_loss(recon_images, images)
            loss = recon_loss + quanitsed_loss
            loss.backward()
            optimiser.step()        

        # Save model
        
        if last_saved >= 10:
            try:
                torch.save(model.state_dict(), model_path)
                last_saved = 0
            except:
                print(f"\nFailed to save at epoch: {epoch}")
        else:
            last_saved += 1
        
        # Infrequent backups
        if epoch > 0 and epoch % 25 == 0:
            try:
                torch.save(model.state_dict(), join(root, "models", f"BACKUP{epoch}-{model_name}"))
            except:
                print(f"\nFailed to save backup at: {epoch}")
        torch.cuda.empty_cache()
    
    
    
def train_audio_vqvae(model_name, load=False, epochs=500):
    
    # Load data for training
    dataset = CustomAudioFolder(join(root,"data/spectrograms"))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=False)
    
    # Create and load model
    model_path = join(root, "models", model_name)
    model = VQVAE()
    if(load):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Establish optimisers
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = torch.nn.MSELoss()
    
    last_saved = 0
    
    # Training
    for epoch in range(epochs):
        for i, audio in enumerate(dataloader):
            print_progress_bar(epoch, i, len(dataloader))
            
            # Convert images for pytorch
            audio = audio.to(device)
        
            # Actual training    
            optimiser.zero_grad()
            recon_audio, quanitsed_loss = model(audio) 
               
            recon_loss = mse_loss(recon_audio, audio)
            loss = recon_loss + quanitsed_loss
            loss.backward()
            optimiser.step()        

        # Save model
        
        if last_saved >= 10:
            try:
                torch.save(model.state_dict(), model_path)
                last_saved = 0
            except:
                print(f"\nFailed to save at epoch: {epoch}")
        else:
            last_saved += 1
        
        # Infrequent backups
        if epoch > 0 and epoch % 50 == 0:
            try:
                torch.save(model.state_dict(), join(root, "models", f"BACKUP{epoch}-{model_name}"))
            except:
                print(f"\nFailed to save backup at: {epoch}")
        torch.cuda.empty_cache()
    

if __name__ == "__main__":  
    
    
    while True:
        answer = input("Would you like to train on image [1] or audio [2]?: ")
        if answer in ["1", "2"]:
            train_image = answer == "1"
            break
        print("Invalid input. Please enter 1 or 2.")
        
    if not train_image:
        audio_model = input("What is the name of your audio model?: ")
        if not audio_model.endswith(".pth"):
            audio_model += ".pth"
        train_audio_vqvae(audio_model, False)
    
    else:
        while True:
            answer = input("Train existing [1] or new model [2]? ").strip()
            if answer in ["1", "2"]:
                load = answer == "1"
                break
            print("Invalid input. Please enter 1 or 2.")

        while True:
            name = input("Please enter the name of your model: ").strip()
            if name:
                if not name.endswith(".pth"):
                    name += ".pth"
                break
            print("Model name cannot be empty.")
        
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
            try:
                epochs = int(input("Please enter the number of epochs (max 2000): ").strip())
                if 0 < epochs <= 2000:
                    break
                else:
                    print("Size must be a positive number no greater than 2000.")
            except ValueError:
                print("Invalid input. Please enter a whole number.")


        while True:
            answer = input("Train hierarchical [1], VQVAE [2], or VAE [3]? ").strip()
            if answer == "1":
                train_vqvae(name, model_type=VQVAE2, load=load, epochs=epochs, image_size=size) 
                break
            elif answer == "2":
                train_vqvae(name, model_type=VQVAE, load=load, epochs=epochs, image_size=size) 
                break
            elif answer == "3":
                train_vae(name, load=load, epochs=epochs, image_size=size)
                break
            else:
                print("Invalid input. Please enter 1, 2, or 3.")
            
    print("Completed training...")
