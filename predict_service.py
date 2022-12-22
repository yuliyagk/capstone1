# -*- coding: utf-8 -*-
import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input

import urllib as ul
import io as io
from PIL import Image

import json

from flask import Flask
from flask import request
from flask import jsonify


url = 'file://./data/images/4138.jpg'

model = keras.models.load_model('xception_v1_04_0.868.h5')

classes = [
    'cups',
    'glasses',
    'plates',
    'spoons',
    'forks',
    'knifes'
]

def prediction_from_url(url) :
   with ul.request.urlopen(url) as requ_obj:
       buffer = requ_obj.read() 
   stream = io.BytesIO(buffer)
   img = Image.open(stream)
   img = img.resize((150, 150))
   x = np.array(img, dtype='float32')
   X = np.array([x])
   X = preprocess_input(X)
   pred = model.predict(X)
   class_predict = dict(zip(classes, pred[0]))
   print(class_predict)
   # wee neet to convert the floats in the dictionary
   d = {k: float(v) for k, v in class_predict.items()}
   predict = json.dumps(d, indent=4) 
   return predict

app = Flask('kitchenware')

@app.route('/predict', methods=['POST'])
def predict():
   url = request.json["url"]
   print(url)
   predict = prediction_from_url(url)
   return jsonify(predict)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
