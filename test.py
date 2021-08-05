from numpy.lib.function_base import extract
import telebot
import librosa as lb
import soundfile as sf
import numpy as np
import os, glob, pickle

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score  

#bot = telebot.TeleBot("1943460407:AAFtrAwI7_Ji-TlUHVPzsDaCK2y8yvxCHFI")

#@bot.message_handler(content_types=['voice'])
#lo que ejecuta es un proceso en el cual descarga el audio de telegram y lo guarda en la carpeta del proyecto
#def voice_processing(message):
#    file_info = bot.get_file(message.voice.file_id)
#    downloaded_file = bot.download_file(file_info.file_path)
#    with open('audio.wav', 'wb') as new_file:
#        new_file.write(downloaded_file)

#bot.polling()

#tabla de etiquetas para los distintos estados de animos
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
    #se recibe el archivo
    with sf.SoundFile(file_title) as audio_recording:
        #el nombre del audio esta en binario y se transforma 
        #con este codigo esto para que a travez de ese valor
        #se de una valoracion cuando saque los atributos del
        #del audio mas abajo
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
            #lo que retorna es una base numerica con los atributos los cuales se trabaja mas adelante
        return result

def loading_audio_data():
    x = []
    y = []
    #se carga y recorre todos los audios del dataset
    for file in glob.glob("D:\Dadafile\Actor_*\*.wav"):

        file_path=os.path.basename(file)
        emotion = emotion_labels[file_path.split("-")[2]]

        if emotion not in focused_emotion_labels:
            continue
        #los envia para recibir los datos de los sonidos
        feature = audio_features(file, mfcc=True, chroma=True, mel=True)
        
        x.append(feature)
        y.append(emotion)

        #toda esa data se guarda en final_dataset a traves del metodo train_test_split que entrana y testea el modelo
    final_dataset = train_test_split(np.array(x), y, test_size=0.1, random_state=9)
    return final_dataset
  #toda esa informacion se almacena en 4 listas para poder diviidrala y trabajarlas por separado
X_train, X_test, y_train, y_test = loading_audio_data()

#se le da la orden al modelo de usar el metodo MLPClassifier con un tamaño de 100 y 700 iteraciones
model = MLPClassifier(hidden_layer_sizes=(200,), learning_rate='adaptive', max_iter=700)

model.fit(X_train,y_train)
#a un array que tiene nombre prediccion se le aplica el modelo de predicion que lleva como paramatre un porsentage del test

Pkl_Filename = "Detector_de_emocion_por_voz_Modelo.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(model, file)

with open(Pkl_Filename, 'rb') as file:
    Modelo_de_detector_de_emociones = pickle.load(file)

Modelo_de_detector_de_emociones    




y_pred = Modelo_de_detector_de_emociones.predict(X_test)
print(y_pred)


#usando el metodo de accyracy_score podemos enviar como parametros estos resutados y sacar un porsentage de efectividad
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("La precision es: {:.1f}%".format(accuracy*100))
#se detalla el porcentaje de efectividad
print("Prediccion")

#EN esta etapa, comienza la implementacion de del bot del telegram
#el cual adquirimos el token, para darle la instruccion de que trabaje cuando le envia un audio

#lo que se hace luego es almacenar ese audio en un archivo  
file = 'audio.wav'



ans = []
#y enviarle ese archivo como paraetr para que lo analise
#eso se almacena en un arreglo
new_feature = audio_features(file, mfcc=True, chroma=True, mel=True)
ans.append(new_feature)
ans = np.array(ans)

#se lo manda como parametro al modelo a ejecutar
arreglo = model.predict([ans])
print(arreglo)