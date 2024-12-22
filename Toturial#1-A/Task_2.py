from Task_1 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define constants
CLASS1_SIZE = 100
CLASS2_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.02
EPOCHS = 100
TEST_SIZE = 0.25

# Define the means and covariances of the two components
MEAN1 = np.array([0, -1])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([-1, 2])
COV2 = np.array([[1, 0], [0, 1]])

# Generate random points from the two components
X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)

# Combine the points and generate labels
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

# Plot the generated data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated Data')
plt.show()

# Split data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Model parameters
n_features = X_train.shape[1]
n_output = 1

# Initialize weights and biases
W = np.random.randn(n_output, n_features) * 0.1
b = np.zeros(n_output)

# Create nodes
x_node = Input()
y_node = Input()

w_node = Parameter(W)
b_node = Parameter(b)

# Use LinearLayer node to replace Multiply and Addition nodes
linear_node = Linear(w_node, b_node, x_node)
sigmoid = Sigmoid(linear_node)
loss = BCE(y_node, sigmoid)

# Create graph outside the training loop
graph = [x_node, w_node, b_node, linear_node, sigmoid, loss]
trainable = [w_node, b_node]

# Training loop
def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()

def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        if t.gradients[t].shape != t.value.shape:
            t.value -= learning_rate * np.squeeze(t.gradients[t])
        else:
            t.value -= learning_rate * t.gradients[t]

for epoch in range(EPOCHS):
    loss_value = 0
    for i in range(X_train.shape[0]):
        x_node.forward(X_train[i].reshape(-1, 1))
        y_node.forward(y_train[i].reshape(1, -1))

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, LEARNING_RATE)

        loss_value += loss.value

    print(f"Epoch {epoch + 1}, Loss: {loss_value / X_train.shape[0]}")

# Evaluate the model
correct_predictions = 0
for i in range(X_test.shape[0]):
    x_node.forward(X_test[i].reshape(-1, 1))
    forward_pass(graph)

    if sigmoid.value > 0.5:  # Assuming binary classification
        prediction = 1
    else:
        prediction = 0

    if prediction == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plot decision boundary
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
Z = []
for i, j in zip(xx.ravel(), yy.ravel()):
    x_node.forward(np.array([i, j]).reshape(-1, 1))
    forward_pass(graph)
    Z.append(sigmoid.value)
Z = np.array(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()

print("Code execution completed.")
