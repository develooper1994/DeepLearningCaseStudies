"""
http://neuralnetworksanddeeplearning.com
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.

All layers are FC(Fully Connected)
"""

#### Libraries
import numpy as np
from scipy.special import expit


class Network(object):
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
                respective layers of the network.  For example, if the list
                was [2, 3, 1] then it would be a three-layer network, with the
                first layer containing 2 neurons, the second layer 3 neurons,
                and the third layer 1 neuron.  The biases and weights for the
                network are initialized randomly, using a Gaussian
                distribution with mean 0, and variance 1.  Note that the first
                layer is assumed to be an input layer, and by convention we
                won't set any biases for those neurons, since biases are only
                ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(n, 1) for n in sizes[1:]]  # (n x 1)
        self.weights = [np.random.randn(n, m) for m, n in zip(sizes[:-1], sizes[1:])]  # (n[l] x n[l-1])

        # exp: sizes = [784, 30, 10]
        # list(zip(sizes[:-1], sizes[1:])) == [(784, 30), (30, 10)]

    def feedforward(self, a):
        """:return: the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b  # (n x 1)
            a = sigmoid(z)  # (n x 1)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        :param training_data: data to train
        :param epochs: how many times use the same data?
        :param mini_batch_size: mini batch size
        :param eta: Learning Rate
        :param test_data: data to test the network
        :return: Nothing returns
        """

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k: k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {epoch} : {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epochs {epoch} complete")

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # let's get gradient with nabla operator
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # takes partial derivative
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  # b + (delta for b)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]  # w + (delta for w)

        # normalize the learning rate according to batch size; (eta/len(mini_batch))
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in
                       zip(self.biases, nabla_b)]  # b - (eta/len(mini_batch))*db
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in
                        zip(self.weights, nabla_w)]  # w - (eta/len(mini_batch))*dw

    def backprop(self, x, y):
        """ :return: a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Let's begin to take partial derivative and use the chain rule
        activation = x  # first activation always is the input of neuron
        activations = [x]  # store all activated neurons layer by layer
        zs = []  # store all z vectors(summed up) layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass, last layer
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)  # delta * transpoze_w[-2]
        # update weights and biases
        # l means layer, I am counting end to start
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sig_deri = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sig_deri  # -2 + 1 = -1
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)  # -2 -1 = -3
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """ test the neural network to answer how many of them are correct. """
        test_results = [
                        (
                            np.argmax(self.feedforward(x)), y
                        )
                        for (x, y) in test_data
                        ]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activation, y):
        """ It actually is a loss function
        :return: the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activation - y


def sigmoid(z):
    """The faster sigmoid function."""
    return expit(z)  # 1.0/(1.0+np.exp(-z))  # slow method


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def relu(Z):
    return np.maximum(0, Z)


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ
