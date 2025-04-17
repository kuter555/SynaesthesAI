import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

root = "C:/Users/chwah/Dropbox/Family/Christopher/University/Y3/Year Long Project/SynaesthesAI/"

def generate_spectrogram_plt(folder, id, name):
    
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    y, sr = librosa.load(f"{root}{folder}{id}.mp3")

    width = sr * 10  # 10 seconds worth of audio
    edges = (len(y) - width) // 2
    y_seg = y[edges:-edges] if edges > 0 else y

    # Generate mel spectrogram
    S = librosa.feature.melspectrogram(y=y_seg, sr=sr, fmax=12000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot and save spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=12000, cmap='magma')
    plt.axis('off')  # Optional: remove axes
    plt.tight_layout(pad=0)

    os.makedirs(f"{root}data/spectrograms", exist_ok=True)
    plt.savefig(f"{root}data/spectrograms/{name}.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    

def generate_spectrogram(folder, id, name):

    
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    y, sr = librosa.load(f"{folder}{id}.mp3")
    
    width = sr * 10
    edges = (len(y)-width) // 2
    y_seg = y[edges:-edges]
    
    S = librosa.feature.melspectrogram(y=y_seg, sr=sr, fmax=12000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    np.save(f"../data/spectrograms/{name}.npy", S_dB)  # Save as NumPy array
    
    