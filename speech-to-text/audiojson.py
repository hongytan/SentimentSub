import json

with open('/Users/hongtan/Desktop/DSClub_Project/audio.json', 'r') as f:
    data = json.load(f)

segments = data['segments']

for segment in segments:
    start = segment['start']
    end = segment['end']
    text = segment['text']
    print(type(start))