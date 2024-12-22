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
        self.value = np.sum(-y_true.value * np.log(y_pred.value) - (1 - y_true.value) * np.log(1 - y_pred.value))

    def backward(self):
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = (1 / y_true.value.shape[0]) * (y_pred.value - y_true.value) / (
                    y_pred.value * (1 - y_pred.value))
        self.gradients[y_true] = (1 / y_true.value.shape[0]) * (np.log(y_pred.value) - np.log(1 - y_pred.value))


# Linear Node
class Linear(Node):
    def __init__(self, A, b, x):
        Node.__init__(self, [A, b, x])

    def forward(self):
        A, b, x = self.inputs
        self.value = np.dot(A.value, x.value) + b.value

    def backward(self):
        A, b, x = self.inputs
        grad_output = self.outputs[0].gradients[self]  # Shape should be (output_size, batch_size)

        # Compute gradients
        self.gradients[A] = np.dot(grad_output, x.value.T)  # Gradient w.r.t A (output_size, input_size)
        self.gradients[b] = np.sum(grad_output, axis=1, keepdims=True)  # Gradient w.r.t b (output_size, 1)
        self.gradients[x] = np.dot(A.value.T, grad_output)  # Gradient w.r.t x (input_size, batch_size)


# Function to automatically create graph and trainable lists
def topological_sort(node):
    visited = set()
    order = []

    def visit(n):
        if n not in visited:
            visited.add(n)
            for m in n.inputs:
                visit(m)
            order.append(n)

    visit(node)
    return order




# Example usage of Linear Node
if __name__ == "__main__":
    A = Parameter(np.array([[0.4, -0.1, 0.6, 0.3],
                            [0.2, 0.7, -0.3, 0.1],
                            [0.5, -0.4, 0.2, 0.8]]))
    b = Parameter(np.array([[0.2], [-0.1], [0.3]]))  # Ensure bias is a column vector
    x = Input()

    linear_node = Linear(A, b, x)

    x.forward(np.array([[0.5], [1.5], [2.5], [1.0]]))  # Ensure input is a column vector
    linear_node.forward()
    print("Forward pass output:", linear_node.value)

    linear_node.outputs = [Node()]
    linear_node.outputs[0].gradients[linear_node] = np.array(
        [[0.2], [-0.2], [0.1]])  # Ensure gradient is a column vector
    linear_node.backward()
    print("Gradient with respect to A:", linear_node.gradients[A])
    print("Gradient with respect to b:", linear_node.gradients[b])
    print("Gradient with respect to x:", linear_node.gradients[x])

    # Create the graph and trainable lists automatically
    graph = topological_sort(linear_node)
    trainables = [node for node in graph if isinstance(node, Parameter)]
    print("Graph:", graph)
    print("Trainable parameters:", trainables)
