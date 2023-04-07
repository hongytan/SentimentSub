#https://blog.paperspace.com/automatic-video-subtitles-with-whisper-autocaption/#notebook-demo-walkthrough


## Imports
from __future__ import unicode_literals
from yt_dlp import YoutubeDL
import yt_dlp
from IPython.display import Video
import whisper
import cv2
import pandas as pd
from moviepy.editor import VideoFileClip
import moviepy.editor as mp
from IPython.display import display, Markdown
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
import os

import cv2

def subtitle_video(download, url, aud_opts, vid_opts, model_type, name, audio_file, input_file, output, uploaded_vid = None):
    ## First, this checks if your expermiment name is taken. If not, it will create the directory.
    ## Otherwise, we will be prompted to retry with a new name
    try:
        os.mkdir(f'experiments/{name}')
        print('Starting AutoCaptioning...')
        print(f'Results will be stored in experiments/{name}')
        
    except:
        return print('Choose another folder name! This one already has files in it.')
    
    ## Use audio and video options for youtube-dl if downloading from youtube
    vid_opts['outtmpl'] = f'experiments/{name}/{input_file}'
    aud_opts['outtmpl'] = f'experiments/{name}/{audio_file}'

    URLS = [url]
    if download:
        with YoutubeDL(aud_opts) as ydl:
            ydl.download(url)
        with YoutubeDL(vid_opts) as ydl:
            ydl.download(URLS)
    else:
        # Use local clip if not downloading from youtube
        my_clip = mp.VideoFileClip(uploaded_vid)
        my_clip.audio.write_audiofile(f'experiments/{name}/{audio_file}')

    # Instantiate whisper model using model_type variable
    model = whisper.load_model(model_type)
    
    # Get text from speech for subtitles from audio file
    result = model.transcribe(f'experiments/{name}/{audio_file}', task = 'translate')
    
    # create Subtitle dataframe, and save it
    dict1 = {'start':[], 'end':[], 'text':[]}
    for i in result['segments']:
        dict1['start'].append(int(i['start']))
        dict1['end'].append(int(i['end']))
        dict1['text'].append(i['text'])
    df = pd.DataFrame.from_dict(dict1)
    df.to_csv(f'experiments/{name}/subs.csv')
    vidcap = cv2.VideoCapture(f'experiments/{name}/{input_file}')
    success,image = vidcap.read()
    height = image.shape[0]
    width =image.shape[1]

    # Instantiate MoviePy subtitle generator with TextClip, subtitles, and SubtitlesClip
    generator = lambda txt: TextClip(txt, font='P052-Bold', fontsize=width/50, stroke_width=.7, color='white', stroke_color = 'black', size = (width, height*.25), method='caption')
    # generator = lambda txt: TextClip(txt, color='white', fontsize=20, font='Georgia-Regular',stroke_width=3, method='caption', align='south', size=video.size)
    subs = tuple(zip(tuple(zip(df['start'].values, df['end'].values)), df['text'].values))
    subtitles = SubtitlesClip(subs, generator)
    
    # Ff the file was on youtube, add the captions to the downloaded video
    if download:
        video = VideoFileClip(f'experiments/{name}/{input_file}')
        final = CompositeVideoClip([video, subtitles.set_pos(('center','bottom'))])
        final.write_videofile(f'experiments/{name}/{output}', fps=video.fps, remove_temp=True, codec="libx264", audio_codec="aac")
    else:
        # If the file was a local upload:
        video = VideoFileClip(uploaded_vid)
        final = CompositeVideoClip([video, subtitles.set_pos(('center','bottom'))])
        final.write_videofile(f'experiments/{name}/{output}', fps=video.fps, remove_temp=True, codec="libx264", audio_codec="aac")



