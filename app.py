import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model =tf.keras.models.load_model('rapi_enggak_160_160_softmax.h5',compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(160, 160))
    show_img = image.load_img(img_path, grayscale=False, target_size=(160, 160))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result_persentase = preds.tolist()
        result_persentase = result_persentase[0][0]
        result_persentase = result_persentase * 100
        result_persentase = int(result_persentase)
        ind=np.argmax(preds)
        if preds[0][0] > 0.80:
           preds = f'Rapi dengan persentase rapi {result_persentase}%'
        else:
           preds =f'Tidak Rapi dengan persentase rapi {result_persentase}%'

        print('Prediction:', preds)
        result=preds
        return result
    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)