#https://stackoverflow.com/questions/36667702/adding-subtitles-to-a-movie-using-moviepy

from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
import whisper_timestamped as whisper

#load the video
video = VideoFileClip("./movie-clips/mclovin.mp4").subclip(0,10)

#load the model
model = whisper.load_model("tiny")
result = model.transcribe(video)
print(result["text"])


generator = lambda txt: TextClip(txt, font='Arial', fontsize=24, color='white')
subs = [((0, 4), 'subs1'),
        ((4, 9), 'subs2'),
        ((9, 12), 'subs3'),
        ((12, 16), 'subs4')]

subtitles = SubtitlesClip(subs, generator)

result = CompositeVideoClip([video, subtitles.set_pos(('center','bottom'))])

result.write_videofile("mclovin_subbed.mp4", fps=video.fps, temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")