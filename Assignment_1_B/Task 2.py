from B_EDF import Input, Parameter, Linear, Sigmoid, BCE
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# XOR dataset generation
CLASS_SIZE = 100  # Number of samples per class

# Class 0: (0, 0) and (1, 1)
# Class 1: (0, 1) and (1, 0)
X_xor = np.vstack([
    np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], CLASS_SIZE),
    np.random.multivariate_normal([1, 1], [[0.1, 0], [0, 0.1]], CLASS_SIZE),
    np.random.multivariate_normal([0, 1], [[0.1, 0], [0, 0.1]], CLASS_SIZE),
    np.random.multivariate_normal([1, 0], [[0.1, 0], [0, 0.1]], CLASS_SIZE)
])
y_xor = np.hstack([
    np.zeros(CLASS_SIZE),  # Class 0
    np.zeros(CLASS_SIZE),  # Class 0
    np.ones(CLASS_SIZE),   # Class 1
    np.ones(CLASS_SIZE)    # Class 1
])

# Plot the XOR dataset
plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('XOR Dataset')
plt.show()

# Split data into training and test sets
np.random.seed(42)
indices = np.random.permutation(len(X_xor))
test_size = int(len(X_xor) * 0.25)

X_train = X_xor[indices[test_size:]]
y_train = y_xor[indices[test_size:]]
X_test = X_xor[indices[:test_size]]
y_test = y_xor[indices[:test_size]]

# MLP parameters
n_features = X_train.shape[1]
hidden_size = 20
output_size = 1

# Create MLP nodes
input_node = Input()
label_node = Input()

# Layer 1
w1 = Parameter(np.random.randn(hidden_size, n_features) * np.sqrt(2 / n_features))
b1 = Parameter(np.zeros((hidden_size, 1)))
layer1 = Linear(w1, b1, input_node)
activation1 = Sigmoid(layer1)

# Layer 2
w2 = Parameter(np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / hidden_size))
b2 = Parameter(np.zeros((hidden_size, 1)))
layer2 = Linear(w2, b2, activation1)
activation2 = Sigmoid(layer2)

# Output Layer
w3 = Parameter(np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size))
b3 = Parameter(np.zeros((output_size, 1)))
output_layer = Linear(w3, b3, activation2)
output_activation = Sigmoid(output_layer)

# Loss Node
loss_node = BCE(label_node, output_activation)

# Graphs for training and visualization
training_graph = [input_node, label_node, w1, b1, layer1, activation1,
                  w2, b2, layer2, activation2, w3, b3, output_layer,
                  output_activation, loss_node]
visualization_graph = [input_node, w1, b1, layer1, activation1,
                       w2, b2, layer2, activation2, w3, b3,
                       output_layer, output_activation]

# Trainable parameters
trainables = [w1, b1, w2, b2, w3, b3]

# Forward and Backward Pass
def forward_pass(graph):
    for node in graph:
        node.forward()

def backward_pass(graph):
    for node in reversed(graph):
        node.backward()

# SGD Update
def sgd_update(trainables, batch_size, learning_rate=1e-2):
    for t in trainables:
        grad_sum = np.sum(t.gradients[t], axis=0)  # Sum gradients over batch
        grad_mean = grad_sum / batch_size  # Compute mean gradient
        t.value -= learning_rate * grad_mean  # Update parameters

# Training loop
epochs = 500
learning_rate = 0.1
BATCH_SIZE = 16

for epoch in range(epochs):
    loss_value = 0
    for i in range(0, X_train.shape[0], BATCH_SIZE):
        x_batch = X_train[i:i + BATCH_SIZE].T  # Shape: (n_features, batch_size)
        y_batch = y_train[i:i + BATCH_SIZE].reshape(1, -1)  # Shape: (1, batch_size)

        input_node.value = x_batch
        label_node.value = y_batch

        forward_pass(training_graph)
        backward_pass(training_graph)
        sgd_update(trainables, BATCH_SIZE, learning_rate)

        loss_value += loss_node.value

    loss_value /= max(X_train.shape[0] // BATCH_SIZE, 1)  # Average loss
    print(f"Epoch {epoch + 1}, Loss: {loss_value:.4f}")

# Evaluate the model
correct_predictions = 0
for i in range(0, X_test.shape[0], BATCH_SIZE):
    x_batch = X_test[i:i + BATCH_SIZE].T
    y_batch = y_test[i:i + BATCH_SIZE].reshape(1, -1)

    input_node.value = x_batch
    forward_pass(visualization_graph)

    predictions = (output_activation.value > 0.5).astype(int)
    correct_predictions += np.sum(predictions == y_batch)

accuracy = correct_predictions / X_test.shape[0]
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Visualize the decision boundary
x_min, x_max = X_xor[:, 0].min() - 0.5, X_xor[:, 0].max() + 0.5
y_min, y_max = X_xor[:, 1].min() - 0.5, X_xor[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()].T  # Shape: (n_features, n_points)

input_node.value = grid
forward_pass(visualization_graph)
Z = output_activation.value.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, cmap='viridis', edgecolor='k')
plt.title('MLP Decision Boundary for XOR Problem')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
