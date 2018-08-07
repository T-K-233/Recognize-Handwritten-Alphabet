from flask import Flask, render_template, request, jsonify
from skimage import transform,data
from matplotlib import pyplot as plt
import base64
import tflearn
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
# import hwa

def get_model():
    net = tflearn.input_data(shape=[None, 784])                                                       # input_layer
    net = tflearn.fully_connected(net, 1024, activation='relu', regularizer='L2', weight_decay=0.001) # dense1
    net = tflearn.dropout(net, 0.9)                                                                   # dropout1
    net = tflearn.fully_connected(net, 512, activation='relu', regularizer='L2', weight_decay=0.001)  # dense2
    net = tflearn.dropout(net, 0.9)                                                                   # dropout2
    net = tflearn.fully_connected(net, 128, activation='relu', regularizer='L2', weight_decay=0.001)  # dense3
    net = tflearn.dropout(net, 0.9)                                                                   # dropout3
    softmax = tflearn.fully_connected(net, 26, activation='softmax')
    # Regression using SGD with learning rate decay and Top-3 accuracy
    sgd = tflearn.SGD(learning_rate=0.04, lr_decay=0.98, decay_step=1000)
    # top_k = tflearn.metrics.Top_k(3)
    net = tflearn.regression(softmax, optimizer=sgd, metric=tflearn.metrics.Accuracy(),
                             loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0)
    
    model.load('model/model.tfl')
    return model


model = get_model()


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('draw.html')

@app.route("/psot", methods=['GET', 'POST'])
def processor():
    # print(request.form)
    img = base64.b64decode(request.form.get('img')[22:])
    with open('temp.jpg', 'wb') as f:
        f.write(img)
    img = plt.imread('temp.jpg')
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img = transform.resize(img, (28, 28))
    img = img[:, :, 0] * 255
    tf.reset_default_graph()
    tflearn.init_graph()
    pred = model.predict([img.reshape(784)])
    res_arr = []
    for i, n in enumerate(pred[0]):
        res_arr.append([float(n), chr(i + 65)])
    res_arr.sort()
    res_arr.reverse()
    return jsonify({'pred': res_arr})


app.run(host='0.0.0.0', port=80, threaded=False)
