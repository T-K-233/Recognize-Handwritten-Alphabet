import tflearn
import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

train_file = "data/a_z_handwritten.npy"

num_classes = 26

def load_data():
    raw = np.load(train_file)
    np.random.shuffle(raw)
    y_label = raw[:, 0]
    out_y = keras.utils.to_categorical(y_label, num_classes)
    num_images = raw.shape[0]
    out_x = raw[:, 1:785]
    return out_x, out_y


X, Y = load_data()

X, Y = X[:100000], Y[:100000]
testX, testY = X[-1000:], Y[-1000:]
# Building deep neural network
net = tflearn.input_data(shape=[None, 784])                                                       # input_layer
net = tflearn.fully_connected(net, 1024, activation='relu', regularizer='L2', weight_decay=0.001) # dense1
net = tflearn.dropout(net, 0.9)                                                                   # dropout1
net = tflearn.fully_connected(net, 256, activation='relu', regularizer='L2', weight_decay=0.001)  # dense2
net = tflearn.dropout(net, 0.9)                                                                   # dropout2
net = tflearn.fully_connected(net, 128, activation='relu', regularizer='L2', weight_decay=0.001)  # dense3
net = tflearn.dropout(net, 0.9)                                                                   # dropout3
softmax = tflearn.fully_connected(net, 26, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.04, lr_decay=0.98, decay_step=1000)
# top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=tflearn.metrics.Accuracy(),
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=4, validation_set=(testX, testY),
          show_metric=True, run_id="dense_model")

model.save('model.tfl')

preds = model.predict(testX)
for i, pred in enumerate(preds):
    print('--- No.%d out of %d ---' % (i + 1, len(preds)))
    pos = high_n = 0
    for m, n in enumerate(testY[i]):
        if n > high_n:
            pos, high_n = m, n
    print('Actual: ', chr(65 + pos))
    pos = high_n = 0
    for m, n in enumerate(pred):
        if n > high_n:
            pos, high_n = m, n
    print('Predict: ', chr(65 + pos))
    plt.imshow(testX[i].reshape(28, 28) * 255, cmap=plt.cm.gray)
    plt.pause(0.5)
    plt.close()
