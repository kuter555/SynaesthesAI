import os
import torch
from networks import VAE, VQVAE, VQVAE2
from PIL import Image
from utils import (
    deconvolve,
    CustomImageFolder,
    CustomAudioImagePairing,
    CustomAudioFolder,
)
import numpy as np
import traceback
import cv2
from os.path import join

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "C:/Users/chwah/Dropbox/Family/Christopher/University/Y3/Year Long Project/SynaesthesAI"


def test_vqvae(input_model, model_type, image_size):

    model_name = input_model.split("/")[-1].split(".")[0]

    model_path = os.path.join(root, "models", input_model)
    if not os.path.exists(model_path):
        print("Model not found.")
        return

    dataset = CustomImageFolder(f"{root}/data/test_images", image_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True
    )

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

        if model_type == VAE:
            recon_images, _, _ = model(images)
        else:
            recon_images, _ = model(images)

        for i in range(len(recon_images)):
            Image.fromarray(
                (
                    deconvolve(
                        recon_images[i].cpu().detach().numpy().squeeze()
                    ).transpose(1, 2, 0)
                    * 255
                ).astype(np.uint8)
            ).save(f"{root}/data/outputs/{model_name}.jpeg")


def test_audio_vqvae(root, audio_model, image_model, size=256):

    audio_path = os.path.join(root, "models", audio_model)
    image_path = os.path.join(root, "models", image_model)

    if not os.path.exists(audio_path) or not os.path.exists(image_path):
        print("Model not found.")
        return
    dataset = CustomAudioImagePairing(
        os.path.join(root, "data/downloaded_images"),
        os.path.join(root, "data/spectrograms"),
        size,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=False
    )

    image_vae = VQVAE2()    
    try:
        image_vae.load_state_dict(torch.load(image_path, map_location=device))
    except Exception as e:
        try:
            checkpoint = torch.load(image_path, map_location=device)
            image_vae.load_state_dict(checkpoint["vqgan"])
        except Exception as e:
            print(f"Unable to load model: {e}. Exiting...")
    image_vae.to(device)

    audio_vae = VQVAE2()
    audio_vae.to(device)
    audio_vae.load_state_dict(torch.load(audio_path, map_location=device))

    for i, (spectrograms, _) in enumerate(dataloader):
        audio_t, audio_b, _, _, _ = audio_vae.encode(spectrograms)
        image_output = image_vae.decode(audio_t, audio_b)
        Image.fromarray(
            (
                deconvolve(
                    image_output[0].cpu().detach().numpy().squeeze()
                ).transpose(1, 2, 0)
                * 255
            ).astype(np.uint8)
        ).save(os.path.join(root, f"data/outputs/first_audio_test_{i}.jpeg"))
        


def load_image(path, size=None):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if size is not None:
        image = cv2.resize(image, (size, size))
    return image


def generate_reconstruction_score(true_image, reconstruction):

    err = np.mean((true_image - reconstruction) ** 2)
    return err




def mse(imageA, imageB):
    """Compute the Mean Squared Error between two images."""
    err = np.mean((np.array(imageA, dtype="float") - np.array(imageB, dtype="float")) ** 2)
    return err

def load_grayscale_image(path):
    """Load image and convert to grayscale."""
    return Image.open(path).convert("L")

def calculate_folder_mse(true_dir, recon_dir):
    """Calculate MSE for all matching images in two folders."""
    errors = {}
    for filename in os.listdir(true_dir):
        true_path = os.path.join(true_dir, filename)
        recon_path = os.path.join(recon_dir, filename)
        
        if not os.path.exists(recon_path):
            print(f"Missing in reconstruction folder: {filename}")
            continue
        
        true_img = load_grayscale_image(true_path)
        recon_img = load_grayscale_image(recon_path)

        # Resize images to the same shape if necessary
        if true_img.size != recon_img.size:
            recon_img = recon_img.resize(true_img.size)

        errors[filename] = mse(true_img, recon_img)
    return errors

if __name__ == "__main__":

    # Load the reference image
    #reference_path = os.path.join(root, "data/test_images/theadul_thestro.jpg")
    #beyonce_true = load_image(reference_path, 256)
#
    ## Folder with comparison images
    #comparison_folder = os.path.join(root, "data/outputs/The Strokes")
#
    ## Iterate over each image file in the folder
    #for filename in os.listdir(comparison_folder):
    #    filepath = os.path.join(comparison_folder, filename)
#
    #    # Skip the reference image
    #    if filename == "theadul_thestro.jpg":
    #        continue
#
    #    # Only process image files
    #    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
    #        try:
    #            image = load_image(filepath, 256)
    #            score = generate_reconstruction_score(beyonce_true, image)
    #            print(f"{filename}: Score = {score}")
    #        except Exception as e:
    #            print(f"Error processing {filename}: {e}")
    #            
                
    test_MSE = False
    if test_MSE:
        
        # Paths
        true_folder = join(root, "data/outputs/originals")
        vqgan_folder = join(root, "data/outputs/AUDIO-VQGAN-FT256")
        vqvae_folder = join(root, "data/outputs/AUDIO-VQVAE2-256")
        
        # Compute MSEs
        vqgan_errors = calculate_folder_mse(true_folder, vqgan_folder)
        vqvae_errors = calculate_folder_mse(true_folder, vqvae_folder)
        
        # Print results
        print("MSE for AUDIO-VQGAN:")
        for img, error in vqgan_errors.items():
            print(f"{img}: {error:.4f}")
        
        print("\nMSE for AUDIO-VQVAE:")
        for img, error in vqvae_errors.items():
            print(f"{img}: {error:.4f}")

                

    #while True:
    #    answer = input("Do you want to test image [1] or audio [2]?: ").strip()
    #    if answer in ["1", "2"]:
    #        break
    #    print("Invalid input. Please enter 1 or 2.")

    if True:

        # audio_model = input("What is the name of your audio model?: ")
        # image_model = input("What is the name of your image model?: ")
        test_audio_vqvae(
            root, "AUDIO-VQGAN-FT256.pth", "256x256/VQGAN-FT256.pth"
        )
    else:

        model_files = []
        model_dir = os.path.join(root, "models")

        # Recursively find all .pth files in subdirectories
        for subdir, _, files in os.walk(model_dir):
            for file in files:
                if file.endswith(".pth"):
                    full_path = os.path.relpath(os.path.join(subdir, file), model_dir)
                    full_path = full_path.replace(os.sep, "/")
                    model_files.append(full_path)

        if not model_files:
            print("No .pth model files found in 'models/' directory.")
        else:
            print("Available VAE model files:")
            for f in model_files:
                print(f"  - {f}")

            while True:
                selected = input(
                    "Enter the name of your VAE model from the list above: "
                ).strip()
                break

            if selected == "all":
                for i in range(len(model_files)):
                    print("Testing model: ", model_files[i])
                    selected = model_files[i]
                    size = int(
                        input(
                            "Please enter the size of these images (max 256): "
                        ).strip()
                    )
                    model = input(
                        "Are you testing a VAE [1], VQVAE [2], or VQVAE2 [3]?: "
                    )
                    if model == "-1":
                        continue
                    try:
                        if model == "1":
                            test_vqvae(selected, VAE, size)
                        elif model == "2":
                            test_vqvae(selected, VQVAE, size)
                        elif model == "3":
                            test_vqvae(selected, VQVAE2, size)
                    except:
                        print("Something went wrong...")

            else:
                while True:
                    try:
                        size = int(
                            input(
                                "Please enter the size of your images (max 256): "
                            ).strip()
                        )
                        if 0 < size <= 256:
                            break
                        else:
                            print("Size must be a positive number no greater than 256.")
                    except ValueError:
                        print("Invalid input. Please enter a whole number.")

                while True:
                    model = input(
                        "Are you testing a VAE [1], VQVAE [2], or VQVAE2 [3]?: "
                    )
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
