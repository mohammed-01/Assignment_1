import numpy as ng
from Task_2 import Conv, MaxPooling

#LinearLayer
class Linear:
    def __init__(self, input_size, output_size):
        self.weights = ng.random.randn(output_size, input_size) * ng.sqrt(2 /input_size)
        self.biases = ng.random.randn(output_size) * 0.01  #smallbias

    def forward(self, x):
        return ng.dot(self.weights, x) + self.biases

#ReLU unction
def relu(x):
    return ng.maximum(0, x)

#CNN class
class CNN:
    def __init__(self):
        self.conv1 = Conv(kernel=ng.random.randn(3 , 3) * ng.sqrt(2 /(3 * 3)), bias=0.01, padding=1)
        self.pool1 = MaxPooling(Pool_Size=(2 , 2), stride=(2, 2))

        self.conv2 = Conv(kernel=ng.random.randn(3 , 3) * ng.sqrt(2 /(3 * 3)), bias=0.01, padding=1)
        self.pool2 = MaxPooling(Pool_Size=(2 , 2), stride=(2, 2))

        self.conv3 = Conv(kernel=ng.random.randn(3 , 3) * ng.sqrt(2 /(3 * 3)), bias=0.01, padding=1)
        self.pool3 = MaxPooling(Pool_Size=(2 , 2), stride=(2 , 2))

        self.conv4 = Conv(kernel=ng.random.randn(3 , 3) * ng.sqrt(2 /(3 * 3)), bias=0.01, padding=1)

        # Calculate flattenned size (spell error)
        self.flattenned_size = 4 * 4  #assume input = 32 x 32 after 3 pooling layers

        self.fc = Linear(input_size=self.flattenned_size, output_size=10)

    def forward(self, x):
        x = self.conv1.operation(x)
        x = relu(x)
        x = self.pool1.operation(x)

        x = self.conv2.operation(x)
        x = relu(x)
        x = self.pool2.operation(x)

        x = self.conv3.operation(x)
        x = relu(x)
        x = self.pool3.operation(x)

        x = self.conv4.operation(x)
        x = relu(x)

        x = x.flatten()

        #making sure the flattened size is the same as the Linear layer input size
        if x.shape[0] != self.flattenned_size:
            raise ValueError(f"flattened input size should be {self.flattenned_size}, but we got {x.shape[0]}.")

        x = self.fc.forward(x)

        return x

#Example
if __name__ == "__main__":
    input_image = ng.random.randn(32, 32)

    nig = CNN()

    output = nig.forward(input_image)

    print("Output logits from the CNN:")
    print(output)

    # Softmax to get probabilities
    def softmax(logits):
        exp_logits = ng.exp(logits - ng.max(logits))  # Subtract max for numerical stability
        return exp_logits / ng.sum(exp_logits)

    probabilities = softmax(output)
    print("Probabilities:", probabilities)
