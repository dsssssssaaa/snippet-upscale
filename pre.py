import torch
import torchaudio
import matplotlib.pyplot as plt

# Load audio file
waveform, sample_rate = torchaudio.load("audio_file.wav")

# Compute spectrogram
n_fft = 400
win_length = 400
hop_length = 160
window = torch.hann_window(win_length)
specgram = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)

# Convert complex spectrogram to magnitude spectrogram
magnitude_specgram = torch.sqrt(specgram[..., 0]**2 + specgram[..., 1]**2)

# Plot spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(torch.log(magnitude_specgram.squeeze() + 1e-9).numpy(), aspect='auto', origin='lower')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.show()
