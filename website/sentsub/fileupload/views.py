from django.shortcuts import render
from .models import Video
from .forms import VideoForm
from transformers import pipeline
from ffmpy import FFmpeg
import stable_whisper

def showvideo(request):

    lastvideo = Video.objects.last()

    videofile = lastvideo.videofile

    form = VideoForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()

    if request.method == 'POST' and 'upload' in request.POST:
        colors = {'fear': '#0000ff', 'joy': '#000000', 'anger':'#ff0000', 'sadness':'#00ff00'}
        classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion')

        # Make sure you are in DS_Project directory when running
        srt_file = '/Users/hongtan/Desktop/sentimentsub/website/sentsub/audio.srt'
        mp4_file = '/Users/hongtan/Desktop/sentimentsub/website/sentsub/media/videos/Friends_Joeys_Bad_Birthday_Gift.mp4'

        # os.system(f'stable-ts {mp4_file} -o {output_file} --word_level False --fp16 False -y')
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

        with open(srt_file, 'w') as f:
            f.writelines(lines)

        # command = f'ffmpeg -i {mp4_file} -vf subtitles={srt_file} output_srt.mp4 -y'
        # os.system(command)

        ff = FFmpeg(
            inputs={f'{mp4_file}': None},
            outputs={'output_srt.mp4': f'-vf subtitles={srt_file} -y'}
        )

        ff.run()

    context= {'videofile': videofile, 'form': form,}

    return render(request, 'fileupload/videos.html', context)

    