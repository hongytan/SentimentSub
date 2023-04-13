import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pydub import AudioSegment
import re
from joblib import load

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
    if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result


#RAVDESS-EMOTIONS
emotions={
    '1':'neutral',
    '2':'calm',
    '3':'happy',
    '4':'sad',
    '5':'angry',
    '6':'fearful',
    '7':'disgust',
    '8':'surpised',
}

#all 03 for audio only
#all 02 for speech
#third num is emotion
#fourth --> emotional intensity, 01 is weak, 02 is strong

observed_emotions = ['happy','sad','angry','fearful', 'disgust']


#DataFlair - Load the data and extract features for each sound 


# def load_data(test_size=0.1):
#     x,y=[],[]
#     index = 0
#     for file in glob.iglob(data_path, recursive= True):
#         file_name=os.path.basename(file)
#         #converting stereo audio to mono
#         sound = AudioSegment.from_wav(file)
#         sound = sound.set_channels(1)
#         sound.export(file, format="wav")
#         emotion=emotions[file_name[7]]
#         feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
#         x.append(feature)
#         y.append(emotion)
#         if index > 20:
#           break
#         index += 1
#     return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

def sort_key(file):
    # Extract the number from the filename using a regular expression
    match = re.search(r'\d+', file)
    if match:
        # Convert the matched number to an integer and return it as the sort key
        return int(match.group())
    else:
        # If no number is found in the filename, return 0 as the sort key
        return 0

def load_data(data_path):
    x = []
    for file in sorted(glob.iglob(data_path, recursive=True), key=sort_key):
        file_name=os.path.basename(file)
        print(file_name)
        #converting stereo audio to mono
        sound = AudioSegment.from_wav(file)
        sound = sound.set_channels(1)
        sound.export(file, format="wav")
        feature=extract_feature("output.wav", mfcc=True, chroma=True, mel=True)
        x.append(feature)
    return x

with open("./audio-analysis/SER.pkl", 'rb') as file:
    model = pickle.load(file)
#model = load('./audio-analysis/SER.joblib')
# Assume 'X_test' is the new data you want to make predictions on
predictions = model.predict(load_data('./segmented-audio/*.wav'))
print(predictions)