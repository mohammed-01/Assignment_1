from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from Task_3 import CNN
from MLP import MLP

#preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#rrain MLP
mlp = MLP(input_size=784, hidden_size=128, output_size=10)
X_train_flat = x_train.reshape(x_train.shape[0], -1)
mlp.train(X_train_flat[:1000], y_train[:1000], epochs=10, lr=0.01)

#train CNN
cnn = CNN()
cnn_output = []
for image in x_train[:1000]:
    cnn_output.append(cnn.forward(image))

#evaluate MLP
X_test_flat = x_test.reshape(x_test.shape[0], -1)
mlp_predictions = np.array([mlp.forward(x) for x in X_test_flat])
mlp_accuracy = np.mean(np.argmax(mlp_predictions, axis=1) == np.argmax(y_test, axis=1))

#evaluate CNN
cnn_predictions = np.array([cnn.forward(image) for image in x_test])
cnn_accuracy = np.mean(np.argmax(cnn_predictions, axis=1) == np.argmax(y_test, axis=1))

print(f'MLP Test Accuracy: {mlp_accuracy}')
print(f'CNN Test Accuracy: {cnn_accuracy}')
