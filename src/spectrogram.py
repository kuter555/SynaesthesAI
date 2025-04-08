import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load('data/Alexchilton.mp3')
S = librosa.feature.melspectrogram(y=y, sr=sr, fmax=12000)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=12000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()