from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
import json

with open('/Users/hongtan/Desktop/DSClub_Project/audio.json', 'r') as f:
    data = json.load(f)

segments = data['segments']
timeframe = []
texts = []
for segment in segments:
    start = segment['start']
    end = segment['end']
    text = segment['text']
    timeframe.append((start, end))
    texts.append(text)

generator = lambda txt: TextClip(txt, font='Arial', fontsize=24, color='white')
subs = []
for i in range(len(texts)):
    subs.append((timeframe[i], texts[i]))

# subs = [((0, 4), 'subs1'),
#         ((4, 9), 'subs2'),
#         ((9, 12), 'subs3'),
#         ((12, 16), 'subs4')]

subtitles = SubtitlesClip(subs, generator)

video = VideoFileClip("/Users/hongtan/Desktop/DSClub_Project/speech-to-text/audio-test-files/Friends Joey's Bad Birthday Gift.mp4")
# video = video.resize( (460,720))
result = CompositeVideoClip([video, subtitles.set_pos(('center','bottom'))])

result.write_videofile("output.mp4", fps=video.fps, temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac", bitrate="5000k")