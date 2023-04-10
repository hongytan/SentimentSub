from django.shortcuts import render
from .models import Video, CaptionVideo
from .forms import VideoForm
from transformers import pipeline
from ffmpy import FFmpeg
import stable_whisper
from django.core.files.base import File

def showvideo(request):

    lastvideo = Video.objects.last()
    videofile = lastvideo.videofile

    # Saves the user submitted form
    form = VideoForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()

    context = {'form': form, 'videofile': videofile}

    # When the user presses the upload button, this goes into effect
    if request.method == 'POST' and 'upload' in request.POST:
        colors = {'fear': '#0000ff', 'joy': '#000000', 'anger':'#ff3333', 'sadness':'#00ff00', 'love':'#ff0000'}
        classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion')

        # Get the last video in database
        lastvideo = Video.objects.last()
        videofile = lastvideo.videofile

        # Make sure you are in DS_Project directory when running
        # CHANGE THIS (ONLY WORKS FOR ABSOLUTE PATHS)
        srt_file = '/Users/hongtan/Desktop/sentimentsub/website/sentsub/audio.srt'
        mp4_file = f'/Users/hongtan/Desktop/sentimentsub/website/sentsub/media/{videofile}'

        # Transcribe the mp4 file into text and outputs a srt file
        model = stable_whisper.load_model('base')
        result = model.transcribe(mp4_file, fp16=False)
        result.to_srt_vtt('audio.srt', word_level=False)

        # Read and classify each line of dialogue
        with open(srt_file, 'r') as f:
            lines = f.readlines()
            n = len(lines)
            for i in range(2,n,4):
                label = classifier(lines[i])[0]['label']
                color = colors[label]
                new_line = f'<font color="{color}">' + lines[i] + '</font>\n'
                lines[i] = new_line

        # Write new and colored subtitles into file
        with open(srt_file, 'w') as f:
            f.writelines(lines)

        # Put subtitles on the video and output the new captioned video
        ff = FFmpeg(
            inputs={f'{mp4_file}': None},
            outputs={'output_srt.mp4': f'-vf subtitles={srt_file} -y'}
        )
        ff.run()

    return render(request, 'fileupload/videos.html', context)

    