import numpy as np

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]

                # forward pass
                for layer in self.layers:
                   output = layer.forward_propagation(output)
 
                err += self.loss(y_train[j], output)

                # backward pass
                output_error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    output_error = layer.backward_propagation(output_error, learning_rate)

            err /= samples
            print("epoch %d/%d    error=%f" % (i+1, epochs, err))

    def predict(self, x_test):
        samples = len(x_test)
        result = []
        for j in range(samples):
            output = x_test[j]
            for layer in self.layers:
               output = layer.forward_propagation(output)

            result.append(output)
        return result
