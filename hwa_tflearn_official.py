import tflearn
import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


load_from_model = True

train_file = 'D:\MachineLearning - local\Handwritten Dataset'

num_classes = 26

def load_data():
    try:
        raw = np.load(train_file)
    except FileNotFoundError as e:
        print('No such file called %s' % train_file)
        print('Loading from csv...')
        raw = np.loadtxt(".\\data\\a_z_handwritten.csv", delimiter=',')
        # raw[:, 1:785] = raw[:, 1:785] / 255
        np.save(train_file, raw)
    np.random.shuffle(raw)
    y_label = raw[:, 0]
    out_y = keras.utils.to_categorical(y_label, num_classes)
    num_images = raw.shape[0]
    out_x = raw[:, 1:785]
    return out_x, out_y


X, Y = load_data()
print(X.shape)
print(Y.shape)

plt.imshow(X[2].reshape(28,28), cmap=plt.cm.gray)
plt.pause(5)
plt.imshow(X[5].reshape(28,28), cmap=plt.cm.gray)
plt.pause(5)
plt.imshow(X[0].reshape(28,28), cmap=plt.cm.gray)
plt.pause(5)

testX, testY = X[-1000:], Y[-1000:]
X, Y = X[:80000], Y[:80000]


# Building deep neural network
def build_model():
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
    return model

# Training 
def train(load_from_model):
    model = build_model()
    if load_from_model:
        model.load('model/model.tfl')
    else:
        model.fit(X, Y, n_epoch=4, validation_set=(testX, testY), show_metric=True, run_id="dense_model")
    return model

model = train(True)
preds = model.predict([testX[2]])
pos = n_hi = 0
for i, n in enumerate(testY[2]):
	if n > n_hi:
		pos, n_hi = i, n
print(chr(pos+65))
pos = n_hi = 0
for i, n in enumerate(preds[0]):
	if n > n_hi:
		pos, n_hi = i, n
print(chr(pos+65))
plt.imshow(testX[2].reshape(28, 28), cmap=plt.cm.gray)
plt.pause(2)
plt.close()
