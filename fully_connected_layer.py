import numpy as np

class FCLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(1, output_size)
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        self.input_error = np.dot(output_error, self.weights.T)
        self.d_weights = np.dot(self.input.T, output_error)
        self.d_bias = output_error

        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias
        return self.input_error
