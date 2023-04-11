from django.shortcuts import render
from .models import Video, CaptionVideo
from .forms import VideoForm
from transformers import pipeline
from ffmpy import FFmpeg
import stable_whisper
from django.core.files.storage import FileSystemStorage
import os
from django.conf import settings

def showvideo(request):

    # if request.method == 'POST' and 'upload' in request.POST:

    form = VideoForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()

    # When the user presses the upload button, this goes into effect
    if request.method == 'POST' and 'upload' in request.POST:
        colors = {'fear': '#0000ff', 'joy': '#000000', 'anger':'#FFCCCC', 'sadness':'#00ff00', 'love':'#ff0000'}
        classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion')

        # Get the last video in database
        lastvideo = Video.objects.last()
        videofile = lastvideo.videofile

        input_filename = str(videofile).split('/')[-1]
        input_file_path = os.path.join(settings.MEDIA_ROOT, 'videos', input_filename)
        srt_filepath = os.path.join(settings.MEDIA_ROOT, 'captions', 'audio.srt')
        output_file_name = input_filename + '_captioned.mp4'
        output_file_path = os.path.join(settings.MEDIA_ROOT, 'captioned_videos', output_file_name)

        # Transcribe the mp4 file into text and outputs a srt file
        model = stable_whisper.load_model('base')
        result = model.transcribe(input_file_path, fp16=False)
        result.to_srt_vtt(srt_filepath, word_level=False)

        # Read and classify each line of dialogue
        with open(srt_filepath, 'r') as f:
            lines = f.readlines()
            n = len(lines)
            for i in range(2,n,4):
                label = classifier(lines[i])[0]['label']
                color = colors[label]
                new_line = f'<font color="{color}">' + lines[i] + '</font>\n'
                lines[i] = new_line

        # Write new and colored subtitles into file
        with open(srt_filepath, 'w') as f:
            f.writelines(lines)

        # Put subtitles on the video and output the new captioned video
        ff = FFmpeg(
            inputs={input_file_path: None},
            outputs={output_file_path: f'-vf subtitles={srt_filepath} -y'}
        )
        ff.run()

        fs = FileSystemStorage()
        filename = fs.save(output_file_name, open(output_file_path, 'rb'))
        uploaded_file_url = fs.url(filename)
        return render(request, 'fileupload/videos.html', {'form': form, 'uploaded_file_url': uploaded_file_url})

    return render(request, 'fileupload/videos.html', {'form': form})

# 

    