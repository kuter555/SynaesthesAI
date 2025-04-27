import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from train_vqvae import train_vqvae
from utils import (
    print_progress_bar,
    HierarchicalLatentDataset,
    LatentDataset,
    extract_latent_codes,
    extract_audio_latent_codes_gpt,
    AudioHierarchicalLatentDataset,
    AudioLatentDataset,
)
from networks import (
    VAE,
    VQVAE,
    VQVAE2,
    LatentGPT,
    BottomGPT,
    AudioLatentGPT,
    AudioBottomGPT,
)

import os
from dotenv import load_dotenv

load_dotenv()
root = os.getenv("root")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 1e-3

## All data loading functions rewritten by ChatGPT for legibility
# 27/04/2025
def load_model(model_cls, path, device):
    """Load a model and return it on the correct device."""
    model = model_cls()
    try:
        state_dict = torch.load(path, map_location=device)
        if "vqgan" in state_dict:
            model.load_state_dict(state_dict["vqgan"])
        else:
            model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed loading model from {path}: {e}")
        return None
    return model.to(device)

def load_tensor(path, device):
    """Load a tensor safely, only move to device if it's a Tensor."""
    data = torch.load(path, map_location=device)
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data

def create_folder(path):
    """Create folder if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def maybe_generate_latents(model_name, audio_filename, t_audio_latents, b_audio_latents, size, output_path):
    """Ask user whether to generate latent codes."""
    answer = input("Do you need to generate latent codes? (y/n): ")
    if answer.lower() == "y":
        create_folder(os.path.join(root, "models", "GPT", output_path))
        extract_audio_latent_codes_gpt(model_name, audio_filename, t_audio_latents, b_audio_latents, size, output_path)

def prepare_dataloaders(t_latents, b_latents, t_audio_latents, b_audio_latents, audio_info, batch_size=32):
    """Setup DataLoaders for top and bottom latents."""
    top_seq = t_latents.view(t_latents.size(0), -1)
    bottom_seq = b_latents.view(b_latents.size(0), -1)
    top_audio_seq = t_audio_latents.view(t_audio_latents.size(0), -1)
    bottom_audio_seq = b_audio_latents.view(b_audio_latents.size(0), -1)
    
    t_audio_dataset = AudioLatentDataset(top_audio_seq, audio_info)
    b_audio_dataset = AudioHierarchicalLatentDataset(top_audio_seq, bottom_audio_seq, audio_info)
    t_dataset = LatentDataset(top_seq)
    b_dataset = HierarchicalLatentDataset(top_seq, bottom_seq)
    
    return {
        "t_audio": DataLoader(t_audio_dataset, batch_size=batch_size, shuffle=True),
        "b_audio": DataLoader(b_audio_dataset, batch_size=batch_size, shuffle=True),
        "t": DataLoader(t_dataset, batch_size=batch_size, shuffle=True),
        "b": DataLoader(b_dataset, batch_size=batch_size, shuffle=True),
    }
    

def train_audio_gpt_hierarchical(model_name, t_latents, b_latents, size, num_epochs=1000, load_top=False):
    audio_filename = input("What is your audio model name?: ")
    audio_model_path = os.path.join(root, "models", audio_filename)
    audio_model = load_model(VQVAE, audio_model_path, device)
    if audio_model is None:
        return -1

    output_path = input("What is your desired output path/where are latents stored?: ")
    maybe_generate_latents(model_name, audio_filename, "audio_" + t_latents, "audio_" + b_latents, size, output_path)

    # Load latents
    try:
        t_latents_tensor = load_tensor(os.path.join(root, "models", "GPT", output_path, t_latents), device)
        b_latents_tensor = load_tensor(os.path.join(root, "models", "GPT", output_path,b_latents), device)
        t_audio_tensor = load_tensor(os.path.join(root, "models", "GPT", output_path, "audio_" + t_latents), device)
        b_audio_tensor = load_tensor(os.path.join(root, "models", "GPT", output_path, "audio_" + b_latents), device)
        audio_info = load_tensor(os.path.join(root, "models", "GPT", output_path, "audio.pt"), device)
    except Exception as e:
        print(f"Failed loading latents: {e}")
        return -1

    # Load image model
    model_path = os.path.join(root, "models", model_name)
    vqvae_model = load_model(VQVAE2, model_path, device)
    if vqvae_model is None:
        return -1

    create_folder(os.path.join(root, "models", "GPT", output_path))

    # Setup dataloaders
    loaders = prepare_dataloaders(t_latents_tensor, b_latents_tensor, t_audio_tensor, b_audio_tensor, audio_info)

    # Setup models and optimizers
    vocab_size = vqvae_model.num_embeddings
    t_model = AudioLatentGPT(vocab_size).to(device)
    b_model = AudioBottomGPT(vocab_size).to(device)

    t_optimiser = torch.optim.Adam(t_model.parameters(), lr=lr)
    b_optimiser = torch.optim.Adam(b_model.parameters(), lr=lr)
    last_save=  0
    if load_top == False:
        
        print("Training top GPT...")
        for epoch in range(num_epochs):
            for i, (inputs, target) in enumerate(loaders["t"]):
                print_progress_bar(epoch, i, len(loaders["t"]))
                inputs, target = inputs.to(device).long(), target.to(device).long()
                logits, loss = t_model(inputs)
                loss.backward()
                t_optimiser.step()
                t_optimiser.zero_grad()

            if last_save > 2:
                try:
                    torch.save(
                        t_model.state_dict(),
                        os.path.join(root, "models", "GPT", output_path, "t_gpt.pth"),
                    )
                    last_save = 0
                except:
                    print("Couldn't save top GPT")
            else:
                last_save += 1
            if epoch % 50 == 0:
                try:
                    torch.save(
                        t_model.state_dict(),
                        os.path.join(
                            root,
                            "models",
                            "GPT",
                            output_path,
                            f"BACKUP{epoch}-t_gpt.pth",
                        ),
                    )
                except Exception as e:
                    print(f"Couldn't save top GPT backup: {e}")
            torch.cuda.empty_cache()

    else:
        print("Loading top GPT")
        t_model.load_state_dict(torch.load(os.path.join(root,"models","GPT",output_path,f"BACKUP{0}-t_gpt.pth"), map_location=device))


    last_save = 0
    # train the bottom GPT
    for epoch in range(num_epochs):
        for i, (inputs, target) in enumerate(loaders["b"]):
            print_progress_bar(epoch, i, len(loaders["b"]))

            inputs, target = inputs.to(device), target.to(device)
            logits, loss = b_model(bottom_inputs, top_seq)
            loss.backward()
            b_optimiser.step()
            b_optimiser.zero_grad()

        if last_save > 2:
            try:
                torch.save(
                    b_model.state_dict(),
                    os.path.join(root, "models", "GPT", output_path, "b_gpt.pth"),
                )
                last_save = 0
            except:
                print("Couldn't save bottom GPT")
        else:
            last_save += 1

        if epoch % 50 == 0:
            try:
                torch.save(
                    b_model.state_dict(),
                    os.path.join(
                        root, "models", "GPT", output_path, f"BACKUP{epoch}-b_gpt.pth"
                    ),
                )
            except Exception as e:
                print(f"Couldn't save bottom GPT backup: {e}")

        torch.cuda.empty_cache()
        
    # Traino on audio this time
    last_save = 0
    print("Conditioning top GPT...")
    for epoch in range(num_epochs):
        for i, (inputs, _, audio) in enumerate(loaders["t_audio"]):
            print_progress_bar(epoch, i, len(loaders["t_audio"]))
            inputs, _, audio = inputs.to(device).long(), _.to(device).long(), audio.to(device)
            logits, loss = t_model(inputs, audio)
            loss.backward()
            t_optimiser.step()
            t_optimiser.zero_grad()
        
        if last_save > 2:
            try:
                torch.save(
                    t_model.state_dict(),
                    os.path.join(root, "models", "GPT", output_path, "t_conditioned_gpt.pth"),
                )
                last_save = 0 
            except:
                print("Couldn't save top GPT")
        else:
            last_save += 1        
        
        if epoch % 50 == 0:
            try:
                torch.save(
                    t_model.state_dict(),
                    os.path.join(
                        root,
                        "models",
                        "GPT",
                        output_path,
                        f"BACKUP{epoch}-t_conditioned_gpt.pth",
                    ),
                )
            except Exception as e:
                print(f"Couldn't save top GPT backup: {e}")
        torch.cuda.empty_cache()

    last_save = 0
    # condition the bottom GPT
    for epoch in range(num_epochs):
        for i, (top_seq, bottom_inputs, bottom_targets, audio) in enumerate(loaders["b_audio"]):
            print_progress_bar(epoch, i, len(loaders["b_audio"]))

            bottom_inputs, target, audio = bottom_inputs.to(device), top_seq.to(device), audio.to(device)
            logits, loss = b_model(bottom_inputs, top_seq, audio)
            loss.backward()
            b_optimiser.step()
            b_optimiser.zero_grad()

        if last_save > 20:
            try:
                torch.save(
                    b_model.state_dict(),
                    os.path.join(root, "models", "GPT", output_path, "b_conditioned_gpt.pth"),
                )
                last_save = 0
            except:
                print("Couldn't save bottom GPT")
        else:
            last_save += 1

        if epoch % 50 == 0:
            try:
                torch.save(
                    b_model.state_dict(),
                    os.path.join(
                        root, "models", "GPT", output_path, f"BACKUP{epoch}-b_conditioned_gpt.pth"
                    ),
                )
            except Exception as e:
                print(f"Couldn't save bottom GPT backup: {e}")

        torch.cuda.empty_cache()



# ALLOWS TRAINING OF VQVAE-2 and VQGAN
def train_vanilla_gpt_hierarchical(
    model_name, t_latents, b_latents, size, num_epochs=100, load_top=False
):

    output_path = input("What is your desired output path/where are latents stored?: ")
    answer = input("Do you need to generate latent codes? (y/n): ")
    if answer == "y":
        if not os.path.exists(os.path.join(root, "models", "GPT", output_path)):
            os.makedirs(os.path.join(root, "models", "GPT", output_path))
        extract_latent_codes(model_name, t_latents, b_latents, size, output_path)

    torch.cuda.empty_cache()
    model = VQVAE2()
    try:
        t_path = os.path.join(root, "models", t_latents)
        b_path = os.path.join(root, "models", b_latents)
        top_latents = torch.load(t_path).to(device)
        bottom_latents = torch.load(b_path).to(device)

    except Exception as e:
        print(f"Failed to run: {e} Exiting...")
        return -1

    try:
        model_path = os.path.join(root, "models", model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["vqgan"])
        except Exception as e:
            print(f"Unable to load model: {e}. Exiting...")
    model.to(device)

    if not os.path.exists(os.path.join(root, "models", "GPT", output_path)):
        os.makedirs(os.path.join(root, "models", "GPT", output_path))

    vocab_size = model.num_embeddings
    top_sequence = top_latents.view(top_latents.size(0), -1)
    bottom_sequence = bottom_latents.view(bottom_latents.size(0), -1)

    t_dataset = LatentDataset(top_sequence)
    b_dataset = HierarchicalLatentDataset(top_sequence, bottom_sequence)
    t_dataloader = DataLoader(t_dataset, batch_size=32, shuffle=True)
    b_dataloader = DataLoader(b_dataset, batch_size=32, shuffle=True)

    t_model = LatentGPT(vocab_size=model.num_embeddings).to(device)
    b_model = BottomGPT(vocab_size=model.num_embeddings).to(device)
    t_optimiser = torch.optim.Adam(t_model.parameters(), lr=lr)
    b_optimiser = torch.optim.Adam(b_model.parameters(), lr=lr)

    if not load_top:
        print("Training top GPT...")
        for epoch in range(num_epochs):
            for i, (inputs, target) in enumerate(t_dataloader):
                print_progress_bar(epoch, i, len(t_dataloader))

                # Get items
                inputs, target = inputs.to(device).long(), target.to(device).long()

                # Train LSTM
                logits = t_model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), target.view(-1)
                )

                loss.backward()
                t_optimiser.step()
                t_optimiser.zero_grad()

            try:
                torch.save(
                    t_model.state_dict(),
                    os.path.join(root, "models", "GPT", output_path, "t_gpt.pth"),
                )
            except:
                print("Couldn't save top GPT")

            if epoch % 50 == 0:
                try:
                    torch.save(
                        t_model.state_dict(),
                        os.path.join(
                            root,
                            "models",
                            "GPT",
                            output_path,
                            f"BACKUP{epoch}-t_gpt.pth",
                        ),
                    )
                except Exception as e:
                    print(f"Couldn't save top GPT backup: {e}")
            torch.cuda.empty_cache()
    

    # train the bottom LSTM
    for epoch in range(num_epochs):
        for i, (input, target) in enumerate(b_dataloader):
            print_progress_bar(epoch, i, len(b_dataloader))

            input, target = input.to(device), target.to(device)
            logits = b_model(input)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-100
            )

            loss.backward()
            b_optimiser.step()
            b_optimiser.zero_grad()

        try:
            torch.save(
                b_model.state_dict(),
                os.path.join(root, "models", "GPT", output_path, "b_gpt.pth"),
            )
        except:
            print("Couldn't save bottom GPT")

        if epoch % 50 == 0:
            try:
                torch.save(
                    b_model.state_dict(),
                    os.path.join(
                        root, "models", "GPT", output_path, f"BACKUP{epoch}-b_gpt.pth"
                    ),
                )
            except Exception as e:
                print(f"Couldn't save bottom GPT backup: {e}")

        torch.cuda.empty_cache()


if __name__ == "__main__":

    while True:
        model_name = input("Please enter the name of your model: ").strip()
        if model_name:
            if not model_name.endswith(".pth"):
                model_name += ".pth"
            break
        print("Model name cannot be empty.")

    while True:
        t_latents = input("What is the name of your top latents?: ").strip()
        if t_latents:
            if not t_latents.endswith(".pt"):
                t_latents += ".pt"
            break
        print("Top latents cannot be empty.")

    while True:
        b_latents = input("What is the name of your bottom latents?: ").strip()
        if b_latents:
            if not b_latents.endswith(".pt"):
                b_latents += ".pt"
            break
        print("Bottom latents cannot be empty.")

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
            num_epochs = int(
                input("Enter the number of training epochs (max 2000): ").strip()
            )
            if 0 < num_epochs <= 2000:
                break
            else:
                print("Number must be between 1 and 2000.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    while True:
        answer = input(
            "Would you like to Extract latents and train (1) or Only train the GPT (2), or Decode (3)? > "
        )
        if answer == "1":
            output_path = input("What is your desired output path?: ")
            extract_latent_codes(model_name, t_latents, b_latents, size, output_path)
            train_audio_gpt_hierarchical(model_name, t_latents, b_latents, size, num_epochs, False)
            break

        elif answer == "2":
            while True:
                Load = input("Load existing top [1] or new top [2]? ").strip()
                if Load in ["1", "2"]:
                    Load = Load == "1"
                    break
                print("Invalid input. Please enter 1 or 2.")
            while True:
                model_type = input(
                    "What model? VAE [1], or VQVAE2/VQGAN-FT [2], or non-audio [3]?: "
                ).strip()
                if model_type == "1":
                    # train_audio_lstm_vae(model_name, t_latents, size, num_epochs)
                    break
                elif model_type == "2":
                    train_audio_gpt_hierarchical(
                        model_name, t_latents, b_latents, size, num_epochs, Load
                    )
                    break
                elif model_type == "3":
                    train_vanilla_gpt_hierarchical(
                        model_name, t_latents, b_latents, num_epochs, Load
                    )
                    break
                else:
                    print("Invalid input. Please enter 1, 2, or 3.")
            break

        elif answer == "3":
            break

        else:
            print("Answer must be one of 1, 2, or 3.")
