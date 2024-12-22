import numpy as np

# Base Node class
class Node:
    def __init__(self, inputs=None):
        if inputs is None:
            inputs = []
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.gradients = {}

        for node in inputs:
            node.outputs.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


# Input Node
class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]


# Parameter Node
class Parameter(Node):
    def __init__(self, value):
        Node.__init__(self)
        self.value = value
        self.gradients = {self: 0}  # Initialize with self as key

    def forward(self):
        pass

    def backward(self):
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]

class Multiply(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        x, y = self.inputs
        self.value = x.value * y.value

    def backward(self):
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self] * y.value
        self.gradients[y] = self.outputs[0].gradients[self] * x.value


class Addition(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        x, y = self.inputs
        self.value = x.value + y.value

    def backward(self):
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self]
        self.gradients[y] = self.outputs[0].gradients[self]


# Sigmoid Activation Node
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        input_value = self.inputs[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        partial = self.value * (1 - self.value)
        self.gradients[self.inputs[0]] = partial * self.outputs[0].gradients[self]


class BCE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        # Clip values to prevent log(0)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred.value, epsilon, 1 - epsilon)
        self.value = np.mean(
            -y_true.value * np.log(y_pred_clipped) - (1 - y_true.value) * np.log(1 - y_pred_clipped)
        )

    def backward(self):
        y_true, y_pred = self.inputs
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred.value, epsilon, 1 - epsilon)
        self.gradients[y_pred] = (y_pred_clipped - y_true.value) / (
            y_pred_clipped * (1 - y_pred_clipped) * y_true.value.shape[0]
        )
        self.gradients[y_true] = -(
            np.log(y_pred_clipped) - np.log(1 - y_pred_clipped)
        ) / y_true.value.shape[0]


# Linear Layer Node
class Linear(Node):
    def __init__(self, W, b, x):
        Node.__init__(self, [W, b, x])
        self.gradients = {W: np.zeros_like(W.value), b: np.zeros_like(b.value), x: np.zeros_like(x.value)}

    def forward(self):
        W, b, x = self.inputs
        self.value = np.dot(W.value, x.value) + b.value

    def backward(self):
        W, b, x = self.inputs
        self.gradients[W] = np.dot(self.outputs[0].gradients[self], x.value.T)
        self.gradients[b] = self.outputs[0].gradients[self].sum(axis=1,
                                                                keepdims=True)  # Ensure it's summed across correct axis
        self.gradients[x] = np.dot(W.value.T, self.outputs[0].gradients[self])

# Example usage of LinearLayer
if __name__ == "__main__":
    # Initialize weights (3x4 matrix) and bias (3,)
    weights = Parameter(np.array([[0.4, -0.1, 0.6, 0.3],
                        [0.2, 0.7, -0.3, 0.1],
                        [0.5, -0.4, 0.2, 0.8]]))
    bias = Parameter(np.array([0.2, -0.1, 0.3]))

    # Create an instance of LinearLayer
    x = Input()
    linear_layer = Linear(weights, bias, x)

    # Example input vector
    x.forward(np.array([0.5, 1.5, 2.5, 1.0]))

    # Perform forward pass
    output = linear_layer.forward()
    print("Forward pass output:", output)

    # Simulate the backward pass with a gradient of the output
    grad_output = np.array([0.2, -0.2, 0.1])  # Gradient of the loss with respect to the output

    # Create an output node (e.g., a placeholder) to connect to the Linear layer
    output_node = Input()
    output_node.value = output  # Set the value of the output node

    # Connect the output node to the linear layer
    linear_layer.outputs.append(output_node)

    # Set the gradient of the output node
    output_node.gradients[linear_layer] = grad_output

    # Perform backward pass
    linear_layer.backward()

    print("Gradient with respect to weights:", weights.gradients[weights])
    print("Gradient with respect to bias:", bias.gradients[bias])