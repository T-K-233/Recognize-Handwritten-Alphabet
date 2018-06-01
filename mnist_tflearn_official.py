import tflearn
import keras
import pandas as pd
# Data loading and preprocessing
import tflearn.datasets.mnist as mnist


train_file = "data/mnist_train.csv"

num_classes = 10

def load_data():
    raw = pd.read_csv(train_file)
    print(raw.label)
    out_y = keras.utils.to_categorical(raw.label, num_classes)
    num_images = raw.shape[0]
    out_x = raw.values[:,1:]
    out_x = out_x / 255
    return out_x, out_y


X, Y, testX, testY = mnist.load_data(one_hot=True)
print('X:', Y.shape)
X, Y = load_data()
print('X:', Y.shape)
testX, testY = X[:100], Y[:100]
# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 784])
dense1 = tflearn.fully_connected(input_layer, 256, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.8)
dense2 = tflearn.fully_connected(dropout1, 128, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)
softmax = tflearn.fully_connected(dropout2, 10, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=20, validation_set=(testX, testY),
          show_metric=True, run_id="dense_model")
