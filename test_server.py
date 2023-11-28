from flask import Flask, request, jsonify
import os

import numpy as np
import librosa
from pydub import AudioSegment, effects
import noisereduce as nr

from tensorflow import keras


app = Flask(__name__)

@app.route('/', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    return jsonify({'message': 'success'}), 200


@app.route('/test')
def hello_world():
    return '<h1>Hello World!</h1><input type="textbox"/>'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

