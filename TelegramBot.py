import librosa
import soundfile
import numpy as np
import glob
import os
import pickle
import telebot

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment

def extract_feature(file_name, **datos):
    mfcc = datos.get("mfcc")
    chroma = datos.get("chroma")
    mel = datos.get("mel")
    contrast = datos.get("contrast")
    tonnetz = datos.get("tonnetz")
    with soundfile.SoundFile(file_name) as archivo_sonido:
        X = archivo_sonido.read(dtype="float32")
        sample_rate = archivo_sonido.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result

# todas las emocines cargadas en el datasetRAVDESS
int2emotion = {
    "01": "neutral",
    "02": "calmado",
    "03": "positivo",
    "04": "triste",
    "05": "negativo",
    "06": "timido",
    "07": "disgustado",
    "08": "sorprendido"
}

# usare las emociones solicitada en el requerimiento
AVAILABLE_EMOTIONS = {
    "negativo",
    "positivo"}

def load_data(test_size=0.2):
    X, y = [], []
    for file in glob.glob("D:\Dadafile\Actor_*\*.wav"):
        # obtengo el nombre base del archivo
        basename = os.path.basename(file)
        # Obtengo la etiqueta de emocion
        emotion = int2emotion[basename.split("-")[2]]
        # permitimos que escoja las etiqueta que necesitamos
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extraccion de caracteristicas
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # se agregan los datos
        X.append(features)
        y.append(emotion)
    # aqui se dividen los datos para el entreamiento
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)
# se carga el test con el 25% de la data
X_train, X_test, y_train, y_test = load_data(test_size=0.25)

# mejor modelo para busqueda
model_params = {
    'alpha': 0.01,
 'batch_size': 256,
    'epsilon': 1e-08,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 700,
}

model = MLPClassifier(**model_params)

# entreanmiento del modelo
print("[*] Entrenando Modelo...")
model.fit(X_train, y_train)

# prediccion con el 25 % de la data
y_pred = model.predict(X_test)
print(y_pred)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print()
print("Efectividad al: {:.2f}%".format(accuracy*100))
print()
print("Tabla de reporte")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
print()

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_pred)
print("Matriz de confusión")
print (matrix)
print()

Pkl_Filename = "Detector_de_emocion_por_voz_Modelo.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(model, file)

with open(Pkl_Filename, 'rb') as file:
    loaded_model = pickle.load(file)

loaded_model

def aextract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result

filename = "03-01-02-01-02-02-21.wav"

new_features = aextract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)

result = loaded_model.predict(new_features)


print("Ejemplo de prueba")
print("Prediccion es : ", result)

bot = telebot.TeleBot("1943460407:AAFtrAwI7_Ji-TlUHVPzsDaCK2y8yvxCHFI")

@bot.message_handler(commands=['seguir'])
def bienvenidaa(message):
    bot.reply_to(message, "Envia un audio" ) 

@bot.message_handler(commands=['preic'])
def bienvenidaa(message):
    bot.reply_to(message, "El estado es positivo" ) 
 
@bot.message_handler(content_types=['voice'])
def voice_processing(update):
    file_info = bot.get_file(update.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('new_file.wav', 'wb') as new_file:
        new_file.write(downloaded_file)
bot.polling() 

