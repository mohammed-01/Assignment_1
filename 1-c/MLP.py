#Ai generated
import numpy as np

# Activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=0)

# MLP Class
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights using He initialization
        self.weights1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size)
        self.biases1 = np.zeros(hidden_size)
        self.weights2 = np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size)
        self.biases2 = np.zeros(output_size)

    def forward(self, x):
        # First layer: Fully connected + ReLU
        self.hidden = relu(np.dot(self.weights1, x) + self.biases1)
        # Second layer: Fully connected + softmax
        output = softmax(np.dot(self.weights2, self.hidden) + self.biases2)
        return output

    def train(self, x_train, y_train, epochs, lr):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                # Forward pass
                self.forward(x)

                # Compute loss (cross-entropy loss)
                loss = -np.sum(y * np.log(self.forward(x) + 1e-9))

                # Backpropagation
                grad_output = self.forward(x) - y
                grad_weights2 = np.outer(grad_output, self.hidden)
                grad_biases2 = grad_output
                grad_hidden = np.dot(self.weights2.T, grad_output) * (self.hidden > 0)
                grad_weights1 = np.outer(grad_hidden, x)
                grad_biases1 = grad_hidden

                # Update weights and biases
                self.weights2 -= lr * grad_weights2
                self.biases2 -= lr * grad_biases2
                self.weights1 -= lr * grad_weights1
                self.biases1 -= lr * grad_biases1

            # Print loss for the epoch
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

# Example usage for MLP
if __name__ == "__main__":
    # Simulated dataset
    x_train = [np.random.randn(64) for _ in range(100)]  # Input size 64
    y_train = np.eye(10)[np.random.randint(0, 10, size=100)]  # One-hot encoded output classes 0-9

    mlp = MLP(input_size=64, hidden_size=128, output_size=10)
    mlp.train(x_train, y_train, epochs=10, lr=0.01)
