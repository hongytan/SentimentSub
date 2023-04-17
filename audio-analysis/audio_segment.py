from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import librosa
import soundfile as sf
import os

dataset_path = '/Users/hongtan/Downloads/archive/audio_speech_actors_01-24'

# Define the data augmentation pipeline
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

for i in range(0,10,1):
    # Augment wav files
    for folder in os.listdir(dataset_path):
        for file in os.listdir(os.path.join(dataset_path, folder)):
            if file.endswith('.wav') and 'augmented' not in file:
                file_path = os.path.join(dataset_path, folder, file)

                # Load the audio file
                samples, sample_rate = librosa.load(file_path, sr=None)

                # Apply data augmentation to the audio samples
                augmented_samples = augment(samples=samples, sample_rate=sample_rate)

                # Save the augmented audio to a new file
                output_file = file_path[:-4] + f'_augmented{i}.wav' 
                sf.write(output_file, augmented_samples, sample_rate) 