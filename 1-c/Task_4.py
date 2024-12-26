import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from Task_3 import CNN

#preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#small subset for quick test
num_samples_train = 100
num_samples_test = 25

x_train = x_train[:num_samples_train]
y_train = y_train[:num_samples_train]
x_test = x_test[:num_samples_test]
y_test = y_test[:num_samples_test]

#train the CNN
cnn = CNN()
num_epochs = 5
batch_size = 10
learning_rate = 0.001
regularization_strength = 0.01
clip_value = 5.0  #gradient clipping value

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Forward pass
        predictions = np.array([cnn.forward(image) for image in x_batch])
        predictions = np.clip(predictions, 1e-10, 1.0)  # Clipping predictions for stability

        # Calculate loss (cross-entropy loss)
        loss = -np.mean(np.sum(y_batch * np.log(predictions), axis=1))

        # Apply L2 regularization
        loss += regularization_strength * (np.sum([np.sum(layer.weights**2) for layer in [cnn.fc1, cnn.fc2]]))

        # Print loss for debugging
        print(f"Batch {i // batch_size + 1}/{len(x_train) // batch_size}, Loss: {loss}")

        # Here, implement the gradients calculation and update cnn's parameters
        # (weights and biases) using gradient descent with learning rate and regularization.
        # Ensure gradient clipping during updates
        # For example:
        # gradients = calculate_gradients(...)
        # gradients = np.clip(gradients, -clip_value, clip_value)
        # update_parameters(cnn, gradients, learning_rate)

    # Evaluate accuracy after each epoch
    y_pred = np.array([cnn.forward(image) for image in x_test])
    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
    print(f"Accuracy after epoch {epoch + 1}: {accuracy:.4f}")

# Softmax function to get probabilities
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # subtract max for numerical stability
    return exp_logits / np.sum(exp_logits)

# Evaluate the final accuracy
cnn_predictions = np.array([cnn.forward(image) for image in x_test])
cnn_probabilities = np.array([softmax(logits) for logits in cnn_predictions])
cnn_accuracy = np.mean(np.argmax(cnn_probabilities, axis=1) == np.argmax(y_test, axis=1))
print(f'Final CNN Test Accuracy: {cnn_accuracy:.4f}')
