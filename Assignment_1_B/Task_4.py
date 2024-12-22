import numpy as np
from sklearn import datasets

# Load the dataset
mnist = datasets.load_digits()
X, y = mnist['data'], mnist['target'].astype(int)

# Ensure no division by zero in normalization (row-wise normalization)
X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-10)

# Manually split the data into training and test sets
np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
train_size = int(0.6 * X.shape[0])
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# One-hot encode the labels
num_classes = len(np.unique(y))
y_train_onehot = np.zeros((y_train.size, num_classes))
y_train_onehot[np.arange(y_train.size), y_train] = 1

# Initialize parameters with Xavier initialization
def initialize_parameters(seed=None):
    np.random.seed(seed)
    input_size = X_train.shape[1]
    hidden_size1 = 256  # Increased hidden size
    hidden_size2 = 128
    output_size = num_classes

    def xavier_init(size):
        in_dim = size[1] if len(size) > 1 else size[0]
        xavier_stddev = np.sqrt(2.0 / in_dim)
        return np.random.randn(*size) * xavier_stddev

    W1 = xavier_init((hidden_size1, input_size))
    b1 = np.zeros((hidden_size1, 1))
    W2 = xavier_init((hidden_size2, hidden_size1))
    b2 = np.zeros((hidden_size2, 1))
    W3 = xavier_init((output_size, hidden_size2))
    b3 = np.zeros((output_size, 1))

    return W1, b1, W2, b2, W3, b3


W1, b1, W2, b2, W3, b3 = initialize_parameters(seed=None)

# Activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)

# Loss function with L2 regularization
def cross_entropy_loss(y_true, y_pred, W1, W2, W3, lambda_reg=0.005):
    epsilon = 1e-10
    cross_entropy = -np.sum(y_true * np.log(y_pred + epsilon)) / y_true.shape[1]
    l2_regularization = lambda_reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2)) / 2
    return cross_entropy + l2_regularization

# Forward pass with dropout
def forward_pass(X, keep_prob=0.8, training=True):
    global dropout_masks
    Z1 = W1.dot(X.T) + b1
    A1 = relu(Z1)
    if training:
        dropout_masks["A1"] = (np.random.rand(*A1.shape) < keep_prob) / keep_prob
        A1 *= dropout_masks["A1"]

    Z2 = W2.dot(A1) + b2
    A2 = relu(Z2)
    if training:
        dropout_masks["A2"] = (np.random.rand(*A2.shape) < keep_prob) / keep_prob
        A2 *= dropout_masks["A2"]

    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# Backward pass and parameter update
def backward_pass(X, y, Z1, A1, Z2, A2, Z3, A3, learning_rate=0.01, lambda_reg=0.005):
    global W1, b1, W2, b2, W3, b3
    m = X.shape[0]

    dZ3 = A3 - y.T
    dW3 = (dZ3.dot(A2.T) + lambda_reg * W3) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    dA2 = W3.T.dot(dZ3)
    dZ2 = dA2 * (Z2 > 0)
    dW2 = (dZ2.dot(A1.T) + lambda_reg * W2) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dA1 = W2.T.dot(dZ2)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = (dZ1.dot(X) + lambda_reg * W1) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

# Learning rate schedule
def adjust_learning_rate(epoch, initial_lr=0.01, decay_rate=0.95, decay_step=10):
    return initial_lr * (decay_rate ** (epoch // decay_step))

# Training
epochs = 300
initial_learning_rate = 0.01
dropout_masks = {}

for epoch in range(epochs):
    learning_rate = adjust_learning_rate(epoch, initial_learning_rate)
    Z1, A1, Z2, A2, Z3, A3 = forward_pass(X_train, training=True)
    loss = cross_entropy_loss(y_train_onehot.T, A3, W1, W2, W3)
    backward_pass(X_train, y_train_onehot, Z1, A1, Z2, A2, Z3, A3, learning_rate=learning_rate)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

# Prediction and evaluation
def predict(X):
    _, _, _, _, _, A3 = forward_pass(X, training=False)
    return np.argmax(A3, axis=0)

y_pred = predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
