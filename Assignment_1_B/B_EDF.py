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

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]


class Multiply(Node):
    def __init__(self, x, y):
        # Initialize with two inputs x and y
        Node.__init__(self, [x, y])

    def forward(self):
        # Perform element-wise multiplication
        x, y = self.inputs
        self.value = x.value * y.value

    def backward(self):
        # Compute gradients for x and y based on the chain rule
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self] * y.value
        self.gradients[y] = self.outputs[0].gradients[self] * x.value


class Addition(Node):
    def __init__(self, x, y):
        # Initialize with two inputs x and y
        Node.__init__(self, [x, y])

    def forward(self):
        # Perform element-wise addition
        x, y = self.inputs
        self.value = x.value + y.value

    def backward(self):
        # The gradient of addition with respect to both inputs is the gradient of the output
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
        self.value = np.sum(-y_true.value * np.log(y_pred.value) - (1 - y_true.value) * np.log(1 - y_pred.value))

    def backward(self):
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = (1 / y_true.value.shape[0]) * (y_pred.value - y_true.value) / (y_pred.value * (1 - y_pred.value))
        self.gradients[y_true] = (1 / y_true.value.shape[0]) * (np.log(y_pred.value) - np.log(1 - y_pred.value))


# Linear Node
class Linear(Node):
    def __init__(self, A, b, x):
        """
        Initialize the Linear node with weight matrix A, bias vector b, and input x.

        Args:
            A (Node): Weight matrix node.
            b (Node): Bias vector node.
            x (Node): Input node.
        """
        Node.__init__(self, [A, b, x])

    def forward(self):
        A, b, x = self.inputs
        self.value = np.dot(A.value, x.value) + b.value

    def backward(self):
        A, b, x = self.inputs
        grad_output = self.outputs[0].gradients[self]  # Shape should be (1, batch_size)

        # Compute gradients
        # Gradient with respect to A: should match A's shape (1, 2)
        self.gradients[A] = np.dot(grad_output, x.value.T)  # This will produce a (1, 2) shape

        # Gradient with respect to b: sum grad_output over the batch dimension
        self.gradients[b] = np.sum(grad_output, axis=1, keepdims=True)

        # Gradient with respect to x
        self.gradients[x] = np.dot(A.value.T, grad_output)  # Shape will be (2, batch_size)


# Example usage of Linear Node
if __name__ == "__main__":
    # Initialize weights (3x4 matrix) and bias (3,)
    A = Parameter(np.array([[0.4, -0.1, 0.6, 0.3],
                            [0.2, 0.7, -0.3, 0.1],
                            [0.5, -0.4, 0.2, 0.8]]))
    b = Parameter(np.array([0.2, -0.1, 0.3]))
    x = Input()

    # Create Linear node
    linear_node = Linear(A, b, x)

    # Example input vector
    x.forward(np.array([0.5, 1.5, 2.5, 1.0]))
    linear_node.forward()
    print("Forward pass output:", linear_node.value)

    # Simulate the backward pass with a gradient of the output
    linear_node.outputs = [Node()]  # Mock output node
    linear_node.outputs[0].gradients[linear_node] = np.array([0.2, -0.2, 0.1])  # Example gradient
    linear_node.backward()
    print("Gradient with respect to A:", linear_node.gradients[A])
    print("Gradient with respect to b:", linear_node.gradients[b])
    print("Gradient with respect to x:", linear_node.gradients[x])
