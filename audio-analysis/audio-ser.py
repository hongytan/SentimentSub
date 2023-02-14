import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
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


emotions={
  'a':'angry',
  'd':'disgust',
  'f':'fearful',
  'h':'happy',
  's':'sad',
}

observed_emotions = ['happy','sad','angry','fearful', 'disgust']


#DataFlair - Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.iglob("C:\\Users\\scott\\Desktop\\Personal\\Personal Projects\\Data Science Proj\\ser-data\\all_data\\**\\*.wav", recursive= True):
        file_name=os.path.basename(file)
        emotion=emotions[file_name[0]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

#split dataset into training set and testing set
x_train,x_test,y_train,y_test=load_data(test_size=0.25)

#print shape and feautures of the dataset
print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted: {x_train.shape[1]}')

model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

#training model
model.fit(x_train,y_train)

#prediciting models for the test set
y_pred=model.predict(x_test)

#assess accuracy of model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: " + str(accuracy*100))