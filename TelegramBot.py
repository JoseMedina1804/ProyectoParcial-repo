
from numpy.lib.function_base import extract
import telebot
import librosa as lb
import soundfile as sf
import numpy as np
import os, glob, pickle

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score  

emotion_labels = {
  '01':'neutral',
  '02':'calmado',
  '03':'positivo',
  '04':'negativo',
  '05':'enojado',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

focused_emotion_labels = ['positivo', 'negativo']

def audio_features(file_title, mfcc, chroma, mel):
    with sf.SoundFile(file_title) as audio_recording:
        audio = audio_recording.read(dtype="float32")
        sample_rate = audio_recording.samplerate
        
        if chroma:
            stft=np.abs(lb.stft(audio))
            result=np.array([])
        if mfcc:
            mfccs=np.mean(lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(lb.feature.melspectrogram(audio, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
        return result

def loading_audio_data():
    x = []
    y = []
    for file in glob.glob("D:\Dadafile\Actor_*\*.wav"):

        file_path=os.path.basename(file)
        emotion = emotion_labels[file_path.split("-")[2]]

        if emotion not in focused_emotion_labels:
            continue
        feature = audio_features(file, mfcc=True, chroma=True, mel=True)
        
        x.append(feature)
        y.append(emotion)

    final_dataset = train_test_split(np.array(x), y, test_size=0.1, random_state=9)
    return final_dataset

X_train, X_test, y_train, y_test = loading_audio_data()


model = MLPClassifier(hidden_layer_sizes=(200,), learning_rate='adaptive', max_iter=700)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("La precision es: {:.1f}%".format(accuracy*100))

print("Prediccion")

bot = telebot.TeleBot("1943460407:AAFtrAwI7_Ji-TlUHVPzsDaCK2y8yvxCHFI")

@bot.message_handler(content_types=['voice'])
def voice_processing(message):
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('audio.wav', 'wb') as new_file:
        new_file.write(downloaded_file)

bot.polling()


file = 'audio.wav'

ans = []
new_feature = audio_features(file, mfcc=True, chroma=True, mel=True)
ans.append(new_feature)
ans = np.array(ans)

model.predict([ans])