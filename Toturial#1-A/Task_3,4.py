import matplotlib.pyplot as plt
from Task_1 import *
import numpy as np
from scipy.stats import multivariate_normal

# Define constants
CLASS1_SIZE = 100
CLASS2_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1
INITIAL_LEARNING_RATE = 0.02
EPOCHS = 100
TEST_SIZE = 0.25
BASE_BATCH_SIZE = 32  # Base batch size for learning rate scaling

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
b = np.zeros((n_output, 1))

# Create nodes
x_node = Input()
y_node = Input()

w_node = Parameter(W)
b_node = Parameter(b)

linear_node = Linear(w_node, b_node, x_node)
sigmoid = Sigmoid(linear_node)
loss = BCE(y_node, sigmoid)

graph = [x_node, w_node, b_node, linear_node, sigmoid, loss]
trainable = [w_node, b_node]


def forward_pass(graph):
    for n in graph:
        n.forward()


def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()


def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        if t.gradients[t].shape != t.value.shape:
            t.gradients[t] = t.gradients[t].reshape(t.value.shape)
        t.value -= learning_rate * t.gradients[t]


# Learning rate scaling factor for observing batch size effect
def adapt_learning_rate(initial_lr, batch_size, base_batch_size=32):
    return initial_lr * (batch_size / base_batch_size)


# Training and evaluation loop with plot
batch_sizes = [1,2,4,8, 32, 64,128,256,512]  # Experiment with various batch sizes
loss_history = {}
accuracy_results = {}

for batch_size in batch_sizes:
    LEARNING_RATE = adapt_learning_rate(INITIAL_LEARNING_RATE, batch_size)

    print(f"\nTraining with batch size: {batch_size} and adjusted learning rate: {LEARNING_RATE}")

    # Re-initialize parameters for each batch size to start fresh
    w_node.value = np.random.randn(n_output, n_features) * 0.1
    b_node.value = np.zeros((n_output, 1))

    losses = []
    for epoch in range(EPOCHS):
        epoch_loss = 0
        indices = np.random.permutation(X_train.shape[0])

        # Iterate over mini-batches
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[indices[i:i + batch_size]].T  # Transpose the batch input
            y_batch = y_train[indices[i:i + batch_size]].reshape(1, -1)

            x_node.forward(X_batch)
            y_node.forward(y_batch)

            forward_pass(graph)
            backward_pass(graph)
            sgd_update(trainable, LEARNING_RATE)

            epoch_loss += loss.value

        average_epoch_loss = epoch_loss / (len(X_train) / batch_size)
        losses.append(average_epoch_loss)
        print(f"Epoch {epoch + 1}, Loss: {average_epoch_loss}")

    loss_history[batch_size] = losses

    # Evaluate the model for this batch size
    correct_predictions = 0
    for i in range(X_test.shape[0]):
        x_node.forward(X_test[i].reshape(-1, 1))
        forward_pass(graph)

        prediction = 1 if sigmoid.value > 0.5 else 0
        if prediction == y_test[i]:
            correct_predictions += 1

    accuracy = correct_predictions / X_test.shape[0]
    accuracy_results[batch_size] = accuracy
    print(f"Accuracy for batch size {batch_size}: {accuracy * 100:.2f}%")

# Plotting Loss History for Different Batch Sizes
for batch_size, losses in loss_history.items():
    plt.plot(range(1, EPOCHS + 1), losses, label=f'Batch Size {batch_size}')

plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Print accuracy results for each batch size
for batch_size, accuracy in accuracy_results.items():
    print(f"Final Accuracy for Batch Size {batch_size}: {accuracy:.2%}")
