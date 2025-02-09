import sys
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image

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
                    self.image_files.append(os.path.join(root, file))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")  # Open image and ensure it's RGB
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # We don't need labels for VAE (return dummy label 0)


def print_progress_bar(epoch, iteration, total, length=50):
    progress = int(length * iteration / total)
    if epoch != -1:
        bar = f"\033[31m Epoch {epoch}:\033[97m [{'=' * progress}{' ' * (length - progress)}] {int(100 * iteration / total)}%"
    else:
        bar = f"[{'=' * progress}{' ' * (length - progress)}] {int(100 * iteration / total)}%"
    sys.stdout.write(f"\r{bar}")
    sys.stdout.flush()