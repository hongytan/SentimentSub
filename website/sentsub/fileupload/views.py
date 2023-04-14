from django.shortcuts import render
from .models import Video, CaptionVideo
from .forms import VideoForm
from transformers import pipeline
from ffmpy import FFmpeg
import stable_whisper
from django.core.files.storage import FileSystemStorage
import os
from django.conf import settings

print("Before classifier")
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion')
print("After classifier")

print("Before model")
model = stable_whisper.load_model('base')
print("After model")


def showvideo(request):

    form = VideoForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()

    # Get the last video uploaded
    if request.method == 'POST' and 'upload' in request.POST:
        colors = {'fear': '#0000ff', 'joy': '#000000', 'anger':'#FFCCCC', 'sadness':'#00ff00', 'love':'#ff0000'}

        # Get the last video uploaded
        lastvideo = Video.objects.last()
        videofile = lastvideo.videofile

        # Get the file paths
        input_filename = str(videofile).split('/')[-1]
        input_file_path = os.path.join(settings.MEDIA_ROOT, 'videos', input_filename)
        srt_filepath = os.path.join(settings.MEDIA_ROOT, 'captions', 'audio.srt')
        output_file_name = input_filename + '_captioned.mp4'
        output_file_path = os.path.join(settings.MEDIA_ROOT, 'captioned_videos', output_file_name)

        # Transcribe the audio
        result = model.transcribe(input_file_path, fp16=False)
        result.to_srt_vtt(srt_filepath, word_level=False)

        # Read the srt file and add color to the lines
        with open(srt_filepath, 'r') as f:
            lines = f.readlines()
            n = len(lines)
            for i in range(2,n,4):
                label = classifier(lines[i])[0]['label']
                color = colors[label]
                new_line = f'<font color="{color}">' + lines[i] + '</font>\n'
                lines[i] = new_line

        # Write the new lines to the srt file
        with open(srt_filepath, 'w') as f:
            f.writelines(lines)

        # Add the colored subtitles to the video
        ff = FFmpeg(
            inputs={input_file_path: None},
            outputs={output_file_path: f'-vf subtitles={srt_filepath} -y'}
        )
        ff.run()

        # Save the captioned video to the database
        fs = FileSystemStorage()
        filename = fs.save(output_file_name, open(output_file_path, 'rb'))
        uploaded_file_url = fs.url(filename)
        return render(request, 'fileupload/videos.html', {'form': form, 'uploaded_file_url': uploaded_file_url})

    # When the user presses the delete button, this goes into effect
    return render(request, 'fileupload/videos.html', {'form': form})

    