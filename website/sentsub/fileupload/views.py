from django.shortcuts import render
from .models import Video
from .forms import VideoForm
from transformers import pipeline
from ffmpy import FFmpeg
import stable_whisper
from django.core.files.storage import FileSystemStorage
import os
from django.conf import settings
import numpy as np
from pydub import AudioSegment
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

def showvideo(request):

    form = VideoForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()

    # =================================================================================================

    # When the user presses the upload button, this goes into effect
    if request.method == 'POST' and 'upload' in request.POST:

        colors = {'Happy': '#FFFF00', 'Angry':'#FF0000', 'Sad':'#0000FF', 'Disgust':'#00FF00', 'Fearful':'#A020F0', 'Neutral':'#FFFFFF', 'Surprised':'#FFA500'}
        emotions = ['Angry', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

        # =================================================================================================

        # Load text sentiment model
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

        # =================================================================================================

        # CONSTANTS
        X_train_shape = (12672, 170, 1)
        input_shape = (X_train_shape[1], 1)

        # Load audio sentiment model
        audio_model = Sequential()
        audio_model.add(Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=input_shape ,use_bias=False))
        audio_model.add(BatchNormalization())
        audio_model.add(MaxPooling1D())
        audio_model.add(Conv1D(filters=16, kernel_size=3, activation='relu', use_bias=False))
        audio_model.add(BatchNormalization())
        audio_model.add(MaxPooling1D())
        audio_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', use_bias=False))
        audio_model.add(BatchNormalization())
        audio_model.add(MaxPooling1D())
        audio_model.add(Flatten())
        audio_model.add(Dense(units=128, activation='relu', use_bias=False))
        audio_model.add(BatchNormalization())
        audio_model.add(Dropout(0.5))
        audio_model.add(Dense(units=7, activation='softmax'))
        audio_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model with appropriate loss function, optimizer and metrics

        # Load weights
        audio_model.load_weights('/Users/hongtan/Desktop/sentimentsub/final_model/')

        # =================================================================================================

        # Function to extract features from audio file
        def extract_features(file_path):
            audio, sampling_rate = librosa.load(file_path, sr=22050, duration=None)
            features=np.array([])
            stft=np.abs(librosa.stft(audio))
            mfccs=np.mean(librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=30).T, axis=0)
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sampling_rate).T,axis=0)
            mel=np.mean(librosa.feature.melspectrogram(y=audio, sr=sampling_rate).T,axis=0)
            features=np.hstack((mfccs, chroma, mel))
            return features

        # =================================================================================================

        # Get the last video in database
        lastvideo = Video.objects.last()
        videofile = lastvideo.videofile

        # =================================================================================================

        # Set up file paths
        input_filename = str(videofile).split('/')[-1]
        input_file_path = os.path.join(settings.MEDIA_ROOT, 'videos', input_filename)
        srt_filepath = os.path.join(settings.MEDIA_ROOT, 'captions', 'audio.srt')
        output_file_name = input_filename.split('.')[0] + '_captioned.mp4'
        output_file_path = os.path.join(settings.MEDIA_ROOT, 'captioned_videos', output_file_name)
        output_wav_name = input_filename.split('.')[0] + '.wav'
        output_wav_path = os.path.join(settings.MEDIA_ROOT, 'wavs', output_wav_name)
        out_segment_dir = os.path.join(settings.MEDIA_ROOT, 'segmented_wavs/')

        # =================================================================================================

        # Transcribe the mp4 file into text and outputs a srt file
        text_model = stable_whisper.load_model('base')
        result = text_model.transcribe(input_file_path, fp16=False)
        result.to_srt_vtt(srt_filepath, word_level=False)

        # =================================================================================================

        # Segment mp4 file into multiple wav files according to SRT file
        # Load audio file
        sound = AudioSegment.from_file(input_file_path, format="mp4")
        sound = sound.set_channels(1)
        sound.export(output_wav_path, format="wav")
        audio = AudioSegment.from_wav(output_wav_path)

        # Parse SRT file
        with open(srt_filepath, "r") as f:
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
        os.makedirs(out_segment_dir, exist_ok=True)

        # Split audio file and export segments
        for i, (start, end) in enumerate(subtitles):
            segment = audio[start:end]
            segment.export(out_segment_dir + str(i) + ".wav", format="wav")

        # =================================================================================================

        audio_probs = [] # Variable to store audio probabilities

        # Read and classify each line of dialogue with audio sentiment model
        for i in range(len(subtitles)):
            audio_file = out_segment_dir + f'{i}.wav'
            audio_features = extract_features(audio_file)
            audio_features = np.expand_dims(audio_features, axis=1)
            test = np.array([audio_features])
            prediction = audio_model.predict(test)
            audio_probs.append(prediction)

        # =================================================================================================

        # Read and classify each line of dialogue with text sentiment model
        with open(srt_filepath, 'r') as f:
            lines = f.readlines()
            n = len(lines)
            for i in range(2,n,4):
                text_clas = classifier(lines[i])[0]
                text_prob = []
                for i in range(7):
                    prob = text_clas[i]['score']
                    text_prob.append(prob)
                text_prob = np.array(text_prob)

                # Combine text and audio probabilities
                prob = (text_prob + audio_probs[int((i-2)/4)] / 5) / (text_prob + audio_probs[int((i-2)/4)] / 5)
                emotion = emotions[np.argmax(prob)]
                color = colors[emotion]

                # Replace line with colored line
                new_line = f'<font color="{color}">' + lines[i] + '</font>\n'
                lines[i] = new_line

        # =================================================================================================

        # Write new and colored subtitles into file
        with open(srt_filepath, 'w') as f:
            f.writelines(lines)

        # =================================================================================================

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

        # =================================================================================================

    return render(request, 'fileupload/videos.html', {'form': form})
    