import numpy as np

class ActivationLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return output_error * self.activation_prime(self.input)
