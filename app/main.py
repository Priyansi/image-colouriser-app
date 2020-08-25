
from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from app.torch_utils import transform_image, get_prediction
import os

app = Flask(__name__)


@app.route('/')
def render_page():
    return render_template('index.html')


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == '':
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

    try:
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        prediction = get_prediction(tensor)
        data = {'prediction': prediction.tolist()}
        return jsonify(data)

    except:
        return jsonify({'error': 'error during prediction'})


if __name__ == '__main__':
    app.run(debug=False, port=os.getenv('PORT', 5000))
