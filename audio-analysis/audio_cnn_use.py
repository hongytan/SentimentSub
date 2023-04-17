from transformers import pipeline
import os
import numpy as np
from pydub import AudioSegment
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

# CONSTANTS
X_train_shape = (12672, 170, 1)
input_shape = (X_train_shape[1], 1)

# Text sentiment model
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Audio sentiment model
model = Sequential()
model.add(Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=input_shape ,use_bias=False))
model.add(BatchNormalization())
model.add(MaxPooling1D())
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', use_bias=False))
model.add(BatchNormalization())
model.add(MaxPooling1D())
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', use_bias=False))
model.add(BatchNormalization())
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(units=128, activation='relu', use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model with appropriate loss function, optimizer and metrics

# Load weights
model.load_weights('/Users/hongtan/Desktop/sentimentsub/final_model/')

# Load audio file
# sound = AudioSegment.from_file(input_file, format="mp4")
# sound = sound.set_channels(1)
# sound.export(output_wav_file, format="wav")
audio = AudioSegment.from_wav(insane_input_file)

# Parse SRT file
with open(output_srt_file, "r") as f:
    lines = f.readlines()
    subtitles = []
    for line in lines:
        if "-->" in line:
            start, end = line.strip().split(" --> ")
            start = start.split(":")
            end = end.split(":")
            start = int(start[0]) * 3600 + int(start[1]) * 60 + float(start[2].replace(",", "."))
            end = int(end[0]) * 3600 + int(end[1]) * 60 + float(end[2].replace(",", "."))
            subtitles.append((start * 1000, end * 1000))

# Create output directory
os.makedirs(output_folder, exist_ok=True)

# Split audio file and export segments
for i, (start, end) in enumerate(subtitles):
    segment = audio[start:end]
    segment.export(output_folder + str(i) + ".wav", format="wav")