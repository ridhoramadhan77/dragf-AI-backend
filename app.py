from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import tensorflow as tf
from skimage import color

# --- Configuration ---
MODEL_PATH = 'model_mst/best_model_temp.h5'
IMG_SIZE = (224, 224)
HEX_LIST = ['#f6ede4','#f3e7db','#f7ead0','#eadaba','#d7bd96','#a07e56','#825c43','#604134','#3a312a','#292420']
LABELS = [f'mst_{i+1}' for i in range(len(HEX_LIST))]

# --- Model & Utils ---
app = Flask(__name__)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# skin segmentation (YCrCb threshold)
def segment_skin(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0,133,77), (255,173,127))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    skin = cv2.bitwise_and(img, img, mask=mask)
    # white background fill
    bg = np.full_like(img, 255)
    bg_mask = cv2.bitwise_not(mask)
    out = cv2.bitwise_and(skin, skin, mask=mask) + cv2.bitwise_and(bg, bg, mask=bg_mask)
    return out

# preprocessing
def preprocess_image(file_stream):
    # read bytes to numpy
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMG_SIZE)
    img = segment_skin(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


# --- Routes ---
@app.route('/')
def index():
    """Render upload page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receive an image, run inference, and return JSON."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    x = preprocess_image(f.stream)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    label = LABELS[idx]
    confidence = float(preds[idx])
    return jsonify({'skintone': label, 'confidence': f'{confidence:.2f}'})

if __name__ == '__main__':
    app.run(debug=True)
