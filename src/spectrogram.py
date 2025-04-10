import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def generate_spectrogram(folder, id, name):

    y, sr = librosa.load(f"{folder}{id}.mp3")

    width = sr * 10
    edges = (len(y)-width) // 2
    y_seg = y[edges:-edges]
    
    S = librosa.feature.melspectrogram(y=y_seg, sr=sr, fmax=12000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    np.save(f"data/spectrograms/{name}.npy", S_dB)  # Save as NumPy array