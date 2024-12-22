from Task_3_EDF import Input, Parameter, Linear, Sigmoid, BCE, topological_sort
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

# Split data
X_train = X_xor[indices[test_size:]]
y_train = y_xor[indices[test_size:]]
X_test = X_xor[indices[:test_size]]
y_test = y_xor[indices[:test_size]]

# Model parameters
n_features = X_train.shape[1]
n_output = 1
BATCH_SIZE = 10  # Define the batch size

# Initialize weights and biases
W = np.random.randn(n_output, n_features) * np.sqrt(2 / n_features)
b = np.zeros((n_output, 1))  # Ensure bias is a column vector

# Create nodes using the imported classes
x_node = Input()
y_node = Input()
w_node = Parameter(W)
b_node = Parameter(b)

# Use Linear node from new_EDF to replace Multiply and Addition nodes
linear_node = Linear(w_node, b_node, x_node)
sigmoid = Sigmoid(linear_node)
loss = BCE(y_node, sigmoid)

# Create the graph and trainable lists automatically
graph = topological_sort(loss)
trainable = [node for node in graph if isinstance(node, Parameter)]

# Training loop
epochs = 1000
learning_rate = 0.001

# Forward and Backward Pass
def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()

# SGD Update
def sgd_update(trainables, batch_size, learning_rate=1e-2):
    for t in trainables:
        grad_sum = np.sum(t.gradients[t], axis=0)  # Sum gradients over batch
        grad_mean = grad_sum / batch_size  # Compute mean gradient

        # Ensure gradient shape matches parameter shape
        if grad_mean.shape != t.value.shape:
            grad_mean = grad_mean.reshape(t.value.shape)

        t.value -= learning_rate * grad_mean  # Update parameters

# Training loop
for epoch in range(epochs):
    loss_value = 0
    for i in range(0, X_train.shape[0], BATCH_SIZE):
        x_batch = X_train[i:i + BATCH_SIZE].T  # Transpose to match the expected shape
        y_batch = y_train[i:i + BATCH_SIZE].reshape(1, -1)

        x_node.value = x_batch
        y_node.value = y_batch

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, BATCH_SIZE, learning_rate)

        loss_value += loss.value

    num_batches = max(X_train.shape[0] // BATCH_SIZE, 1)  # Ensure at least 1 batch
    loss_value = loss_value / num_batches
    print(f"Epoch {epoch + 1}, Loss: {loss_value}")

# Evaluate the model
correct_predictions = 0
for i in range(0, X_test.shape[0], BATCH_SIZE):
    x_batch = X_test[i:i + BATCH_SIZE].T  # Shape: (n_features, batch_size)
    y_batch = y_test[i:i + BATCH_SIZE].reshape(1, -1)  # Shape: (1, batch_size)

    x_node.value = x_batch
    y_node.value = y_batch

    forward_pass(graph)

    predicted = (sigmoid.value > 0.5).astype(int)  # Shape: (1, batch_size)

    # Ensure broadcasting works by slicing `predicted` or `y_batch` if needed
    correct_predictions += np.sum(predicted[:, :y_batch.shape[1]] == y_batch)

accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the decision boundary
x_min, x_max = X_xor[:, 0].min(), X_xor[:, 0].max()
y_min, y_max = X_xor[:, 1].min(), X_xor[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Predict the grid points
Z = np.zeros_like(xx.ravel())
for idx, (i, j) in enumerate(zip(xx.ravel(), yy.ravel())):
    x_node.value = np.array([[i], [j]])  # Adjust shape to (2, 1)
    forward_pass(graph)
    Z[idx] = sigmoid.value.item()  # Ensure we get the scalar value

Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, cmap='viridis', edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary for XOR')
plt.show()
