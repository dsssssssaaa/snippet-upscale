import librosa
import numpy as np
import os

def preprocess_and_save_audio_data(data_dir, target_sr=22050, duration=10):
    # Form paths to the 'dataHQ' and 'dataLQ' subfolders
    hq_dir = os.path.join(data_dir, 'dataHQ')
    lq_dir = os.path.join(data_dir, 'dataLQ')

    # Process audio files from 'dataHQ' subfolder
    for file in os.listdir(hq_dir):
        file_path = os.path.join(hq_dir, file)
        if file.endswith('.wav'):  # Check if the file is a WAV file
            processed_audio = preprocess_audio(file_path, target_sr=target_sr, duration=duration)
            save_path = os.path.join(hq_dir, 'hqpro', file.split('.')[0] + '.npy')
            np.save(save_path, processed_audio)

    # Process audio files from 'dataLQ' subfolder
    for file in os.listdir(lq_dir):
        file_path = os.path.join(lq_dir, file)
        if file.endswith('.wav'):  # Check if the file is a WAV file
            processed_audio = preprocess_audio(file_path, target_sr=target_sr, duration=duration)
            save_path = os.path.join(lq_dir, 'lqpro', file.split('.')[0] + '.npy')
            np.save(save_path, processed_audio)

def preprocess_audio(file_path, target_sr=22050, duration=10):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=target_sr, duration=duration, mono=True)

    # Extract features (e.g., Mel spectrogram)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

    # Convert amplitude to decibels (log scale)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Normalize spectrogram values between 0 and 1
    mel_spectrogram_db_normalized = (mel_spectrogram_db - np.min(mel_spectrogram_db)) / (np.max(mel_spectrogram_db) - np.min(mel_spectrogram_db))

    return mel_spectrogram_db_normalized

# Example usage:
data_dir = "/content/snippet-upscale/data"
preprocess_and_save_audio_data(data_dir)
