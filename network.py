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
        err = 0
        for epoch in range(1, epochs):
            for i, data in enumerate(x_train):
                output = data
                #forward_propagation
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                #Compute the error for display purpuse
                err += self.loss(y_train[i], output)

                #backward_propagation
                output_error = self.loss_prime(y_train[i], output)
                for layer in reversed(self.layers):
                    output_error = layer.backward_propagation(output_error, learning_rate)

                # calculate average error on all samples
            err /= len(x_train)
            print('Epoch %d/%d  Error=%f' % (epoch+1, epochs, err))

    def predict(self, x_test):
        res = []
        for i, data in enumerate(x_test):
            #forward_propagation
            output = data
            for layer in self.layers:
                output = layer.forward_propagation(output)
            res.append(output)
        return res
