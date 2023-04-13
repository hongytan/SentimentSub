from pydub import AudioSegment
import os

# Load audio file
sound = AudioSegment.from_file("./audio-analysis/racist.mp4", format="mp4")
sound = sound.set_channels(1)
sound.export("output.wav", format="wav")
audio = AudioSegment.from_wav("output.wav")

# Parse SRT file
with open("./audio.srt", "r") as f:
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
os.makedirs("segmented-audio", exist_ok=True)

# Split audio file and export segments
for i, (start, end) in enumerate(subtitles):
    segment = audio[start:end]
    segment.export("segmented-audio/" + str(i) + ".wav", format="wav")