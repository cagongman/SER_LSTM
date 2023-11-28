from flask import Flask, request, jsonify
import os
from openai import OpenAI

import numpy as np
import librosa
from pydub import AudioSegment, effects
import noisereduce as nr

from tensorflow import keras


# Emotions list is created for a readable form of the model prediction.
emotions = {
    0 : 'neutral',
    1 : 'happy',
    2 : 'sad',
    3 : 'angry',
}
emo_list = list(emotions.values())

model = None


def preprocess(file_path, frame_length = 2048, hop_length = 512):
    '''
    A process to an audio .wav file before execcuting a prediction.
      Arguments:
      - file_path - The system path to the audio file.
      - frame_length - Length of the frame over which to compute the speech features. default: 2048
      - hop_length - Number of samples to advance for each frame. default: 512

      Return:
        'X_3D' variable, containing a shape of: (batch, timesteps, feature) for a single file (batch = 1).
    '''
    total_length = 173056
 
    # Fetch sample rate.
    _, sr = librosa.load(path = file_path, sr = None)
    # Load audio file
    rawsound = AudioSegment.from_file(file_path, duration = None) 
    # Normalize to 5 dBFS 
    normalizedsound = effects.normalize(rawsound, headroom = 5.0) 
    # Transform the audio file to np.array of samples
    normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32')
	
    # Trim silence from the beginning and the end.
    xt, index = librosa.effects.trim(normal_x, top_db=30)
    #print(file,"\t", len(xt), "\t", rawsound.dBFS, "\t", normalizedsound.dBFS) #--QA purposes if needed--
    # Pad for duration equalization.
    # if total_length - len(xt) < 0:
    #     print("over" + str(total_length - len(xt)))
    #     continue
    padded_x = np.pad(xt, (0, total_length-len(xt)), 'constant') 
    
    # Noise reduction                  
    # final_x = nr.reduce_noise(normal_x, sr=sr, use_tensorflow=True)
    final_x = nr.reduce_noise(padded_x, sr=sr)
        
        
    f1 = librosa.feature.rms(y=final_x, frame_length=frame_length, hop_length=hop_length, center=True, pad_mode='reflect').T # Energy - Root Mean Square
    f2 = librosa.feature.zero_crossing_rate(y=final_x, frame_length=frame_length, hop_length=hop_length,center=True).T # ZCR
    f3 = librosa.feature.mfcc(y=final_x, sr=sr, S=None, n_mfcc=13, hop_length = hop_length).T # MFCC   
    X = np.concatenate((f1, f2, f3), axis = 1)
    
    X_3D = np.expand_dims(X, axis=0)
    
    return X_3D


def predict_emotion(filePath):
    global model

    # Initialize variables
    RATE = 24414
    CHUNK = 512
    RECORD_SECONDS = 7.1

    CHANNELS = 1
    AUDIO_FILE_PATH = filePath

    x = preprocess(AUDIO_FILE_PATH) # 'output.wav' file preprocessing.

    predictions = model.predict(x)
    print('Nuetral: ' + str(predictions[0][0]))
    print('Happy: ' + str(predictions[0][1]))
    print('Sad: ' + str(predictions[0][2]))
    print('Angry: ' + str(predictions[0][3]))
    max_emo = np.argmax(predictions)
    #print(max_emo)
    print('max emotion:', emotions.get(max_emo,-1))


def load_model():
    global model

    saved_model_path = './model8723.json'
    saved_weights_path = './model8723_weights.h5'

    #Reading the model from JSON file
    with open(saved_model_path, 'r') as json_file:
        json_savedModel = json_file.read()

    # Loading the model architecture, weights
    model = keras.models.model_from_json(json_savedModel)
    model.load_weights(saved_weights_path)

    # Compiling the model with similar parameters as the original model.
    model.compile(loss='categorical_crossentropy',
                    optimizer='RMSProp',
                    metrics=['categorical_accuracy'])

    print(model.summary())


def is_silent(data):
    # Returns 'True' if below the 'silent' threshold
    return max(data) < 100



app = Flask(__name__)

@app.route('/', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    filePath = './audio/' + file.filename 
    file.save(filePath)
    print('File successfully uploaded -- ' + file.filename)

    predict_emotion(filePath)
    
    return jsonify({'message': 'success'}), 200

@app.route('/test')
def hello_world():
    return '<h1>Hello World!</h1><input type="textbox"/>'

if __name__ == '__main__':
    # load SER model and print model summary
    load_model()
    # run server!!
    app.run(host='0.0.0.0', port=5000)
