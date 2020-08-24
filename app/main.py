
from flask import Flask, request, jsonify
import os
import numpy as np
import torch.utils.py

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == '':
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

    print('here')

    # try:
    img_bytes = file.read()
    tensor = transform_image(img_bytes)
    prediction = get_prediction(tensor)
    data = {'prediction': prediction}
    return jsonify(data)

    # except:
    # return jsonify({'error': 'error during prediction'})

    return jsonify({'result': 1})
