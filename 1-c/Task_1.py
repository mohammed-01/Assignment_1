import numpy as ng
import matplotlib.pyplot as plt
from keras.datasets import mnist

#Loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#Print the shapes of the vectors
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
plt.show()

# data processing
train_X = train_X.reshape(train_X.shape[0], -1) / 255.0  # Normalize to [0, 1]
test_X = test_X.reshape(test_X.shape[0], -1) / 255.0

#encoding the labels
num_classes = 10
train_y = ng.eye(num_classes)[train_y]
test_y = ng.eye(num_classes)[test_y]

#functions
def relu(x):
    return ng.maximum(0, x)

def softmax(x):
    exp_values = ng.exp(x - ng.max(x, axis=1, keepdims=True))
    return exp_values / ng.sum(exp_values, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -ng.log(y_pred[range(m), y_true.argmax(axis=1)])
    loss = ng.sum(log_likelihood) / m
    return loss

#forward and backward pass functions
def forward_pass(X, params):
    cache = {}
    cache['A0'] = X
    for l in range(1, len(params) // 2 + 1):
        Z = ng.dot(cache[f'A{l-1}'], params[f'W{l}']) + params[f'b{l}']
        cache[f'Z{l}'] = Z
        if l == len(params) // 2:
            cache[f'A{l}'] = softmax(Z)
        else:
            cache[f'A{l}'] = relu(Z)
    return cache

def backward_pass(y, params, cache):
    gradients = {}
    L = len(params) // 2
    m = y.shape[0]

    da = cache[f'A{L}'] - y
    for l in reversed(range(1, L + 1)):
        dz = da * (cache[f'Z{l}'] > 0) if l < L else da
        gradients[f'dW{l}'] = ng.dot(cache[f'A{l-1}'].T, dz) / m
        gradients[f'db{l}'] = ng.sum(dz, axis=0, keepdims=True) / m
        if l > 1:
            da = ng.dot(dz, params[f'W{l}'].T)

    return gradients

#Initialize parameters with uniform initializaton
def initialize_parameters(layer_dims):
    params = {}
    ng.random.seed(1)

    for l in range(1, len(layer_dims)):
        params[f'W{l}'] = ng.random.uniform(-1, 1, (layer_dims[l-1], layer_dims[l]))
        params[f'b{l}'] = ng.zeros((1, layer_dims[l]))

    return params

#Updating parameters using L2 regularization
def update_parameters(params, gradients, learning_rate, lambda_reg, m):
    for l in range(1, len(params) // 2 + 1):
        params[f'W{l}'] -= learning_rate * (gradients[f'dW{l}'] + (lambda_reg / m) * params[f'W{l}'])
        params[f'b{l}'] -= learning_rate * gradients[f'db{l}']
    return params

#model training function
def train_model(X, y, layer_dims, learning_rate, epochs, lambda_reg):
    params = initialize_parameters(layer_dims)
    for epoch in range(epochs):
        cache = forward_pass(X, params)
        gradients = backward_pass(y, params, cache)
        params = update_parameters(params, gradients, learning_rate, lambda_reg, X.shape[0])

        if (epoch + 1) % 10 == 0:
            loss = cross_entropy_loss(y, cache[f'A{len(layer_dims) - 1}'])
            print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

    return params

#Predict function
def predict(X, params):
    cache = forward_pass(X, params)
    predictions = ng.argmax(cache[f'A{len(params) // 2}'], axis=1)
    return predictions

#model parameters
input_size = 784  # 28x28 images
hidden_sizes = [128, 64, 32]  # Example: Three hidden layers
output_size = 10  # 10 classes
layer_dims = [input_size] + hidden_sizes + [output_size]

#Hyperparameters
learning_rate = 0.001  # Further reduced learning rate
epochs = 200  # Increased number of epochs
lambda_reg = 0.01  # L2 regularization parameter

#Train the model
params = train_model(train_X, train_y, layer_dims, learning_rate, epochs, lambda_reg)

#Evaluate the model
test_predictions = predict(test_X, params)
test_accuracy = ng.mean(test_predictions == test_y.argmax(axis=1))
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
