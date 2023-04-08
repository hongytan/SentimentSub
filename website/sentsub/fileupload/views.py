from django.shortcuts import render
from .models import Video
from .forms import VideoForm
# from ../text-analysis/pipeline import Upload

def showvideo(request):

    lastvideo = Video.objects.last()

    videofile = lastvideo.videofile

    form = VideoForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()

    print("Hello World")

    context= {'videofile': videofile, 'form': form,}

    return render(request, 'fileupload/videos.html', context)

    