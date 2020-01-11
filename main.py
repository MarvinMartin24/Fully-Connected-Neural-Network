import numpy as np

from network import Network
from fully_connected_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

def main():
    x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    net = Network()
    net.add(FCLayer(2, 3))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(3, 1))
    net.add(ActivationLayer(tanh, tanh_prime))

    net.use(mse, mse_prime)
    net.fit(x_train, y_train, 1000, 0.1)

    out = net.predict(x_train)
    print(out)

if __name__ == '__main__':
    main()
