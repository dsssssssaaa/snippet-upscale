import argparse
from pydub import AudioSegment
import os

def split_audio_folder(input_folder, output_folder, segment_length=5000):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):  # Assuming all files in the folder are WAV format, you can adjust this condition if needed
            # Load audio file
            audio = AudioSegment.from_file(os.path.join(input_folder, filename))

            # Calculate the number of segments
            num_segments = len(audio) // segment_length

            # Split audio into segments
            for i in range(num_segments):
                start_time = i * segment_length
                end_time = (i + 1) * segment_length
                segment = audio[start_time:end_time]
                segment.export(os.path.join(output_folder, f"{filename[:-4]}_segment_{i}.wav"), format="wav")

            # Handle the last segment (which may be shorter than segment_length)
            if len(audio) % segment_length != 0:
                start_time = num_segments * segment_length
                segment = audio[start_time:]
                segment.export(os.path.join(output_folder, f"{filename[:-4]}_segment_{num_segments}.wav"), format="wav")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split audio files into segments.')
    parser.add_argument('--data', help='Input data folder containing audio files', required=True)
    parser.add_argument('--dataoutput', help='Output folder where segmented audio files will be saved', required=True)
    args = parser.parse_args()

    input_folder = args.data
    output_folder = args.dataoutput
    segment_length = 5000  # in milliseconds (5 seconds)

    split_audio_folder(input_folder, output_folder, segment_length)
