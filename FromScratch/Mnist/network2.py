"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""
#### Libraries
# Standard library
import json, random, sys

# Third-party libraries
import numpy as np
from scipy.special import expit


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        """
        Return the cost associated with an output ``a`` and desired output ``y``.
        :param a: activation output
        :param y: desired output
        :return: euclidean (norm or cost)
        """
        return 0.5 * np.linalg.norm(a - y) ** 2  # euclidean (norm or cost)

    @staticmethod
    def delta(z, a, y):
        """
        Return the error delta from the output layer.
        :param z:
        :param a: activation output
        :param y: desired output
        :return: error delta from the output layer. In the other words, it is a derivative of fn
        """
        return (a - y) * sigmoid_prime(z)  # cost_derivative * sigmoid_prime


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """
        Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        :param a: activation output
        :param y: desired output
        :return: cross entropy
        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """
        Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        :param z:
        :param a: activation output
        :param y: desired output
        :return:
        """
        return a - y


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).
        :param sizes:
        :param cost:
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """
        Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        :return: Nothing
        """

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]  # (n[l] x n[l-1])

        # exp: sizes = [784, 30, 10]
        # list(zip(sizes[:-1], sizes[1:])) == [(784, 30), (30, 10)]

    def large_weight_initializer(self):
        # for large weight values.
        """
        Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.
        :return: Nothing
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]  # (n[l] x n[l-1])

        # exp: sizes = [784, 30, 10]
        # list(zip(sizes[:-1], sizes[1:])) == [(784, 30), (30, 10)]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b  # (n x 1)
            a = sigmoid(z)  # (n x 1)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0,
            evaluation_data=None, monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False, monitor_training_cost=False,
            monitor_training_accuracy=False, early_stopping_n=0):
        """
        Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.
        :param training_data: data to train
        :param epochs: how many times use the same data?
        :param mini_batch_size: mini batch size
        :param eta: Learning Rate
        :param lmbda: non-optional parameters are self-explanatory, as is the regularization parameter
        :param evaluation_data: validation or test
        :param monitor_evaluation_cost:
        :param monitor_evaluation_accuracy:
        :param monitor_training_cost:
        :param monitor_training_accuracy:
        :param early_stopping_n:
        :return: Nothing returns
        """

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(training_data)
            n_data = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k: k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))

            print(f"Training {epoch}. epoch completed")

            # Monitor the results(cost and accuracy)
            evaluation_cost, evaluation_accuracy = [], []
            training_cost, training_accuracy = [], []
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"Cost on training data : {cost}")
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data : {accuracy} / {n}")

            if monitor_evaluation_cost:
                ev_cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(ev_cost)
                print(f"Cost on evaluation data : {ev_cost}")
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {self.accuracy(evaluation_data)} / {n_data}")

        # # early stopping mechanism
        # best_accuracy = 0
        # no_accuracy_change = 0
        #
        # if early_stopping_n > 0:
        #     if accuracy > best_accuracy:
        #         best_accuracy = accuracy
        #         no_accuracy_change = accuracy
        #         #print("Early-stopping: Best so far {}".format(best_accuracy))
        #     else:
        #         no_accuracy_change += 1
        #
        #     #if no_accuracy_change == early_stopping_n:
        #         #print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))

        # return and finish the SGD
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        :param mini_batch: list of tuples (x, y)
        :param eta: learning rate
        :param lmbda: regularization parameter
        :param n: total size of the training data set
        :return: Nothing
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [
            (1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * dw
            for w, dw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * db
            for b, db in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        :param x: input
        :param y: output
        :return: derivative of biases and weights according to the Cost function
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward pass
        activation = x  # first activation always is the input of neuron
        activations = [x]  # store all activated neurons layer by layer
        zs = []  # store all z vectors(summed up) layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b  # (n x 1)
            zs.append(z)
            activation = sigmoid(z)  # (n x 1)
            activations.append(activation)

        # backward pass(output layer)
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
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
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        return nabla_b, nabla_w





    def accuracy(self, data, convert=False):
        """
        Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.
        :param data: input data
        :param convert: should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data
        :return: accuracy
        """
        if convert:
            results = [
                (
                    np.argmax(self.feedforward(x)), np.argmax(y)
                )
                for x, y in data
            ]
        else:
            results = [
                (
                    np.argmax(self.feedforward(x)), y
                 )
                for x, y in data
            ]
        test = []
        for (x, y) in results:
            test.append(int(x == np.nonzero(y[:])[0]))

        accuracy = sum(test)
        return accuracy











    def total_cost(self, data, lmbda, convert=False):
        """
        Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        :param data: input data
        :param lmbda: regularization rate
        :param convert: should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data
        :return:
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y)
            cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename):
        """
        Save the neural network to the file ``filename``.
        :param filename: neural network file includes weights and biases
        :return: Nothing
        """
        data = {
            "sizes": self.sizes,
            "weight": [list(w) for w in self.weights],
            "bias": [list(b) for b in self.biases],
            "cost": str(self.cost.__name__)
        }
        with open(filename, 'w') as f:
            json.dump(data)


#### Loading a Network
def load(filename):
    """
    Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    :param filename: neural network file includes weights and biases
    :return: loaded weights and biases
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


#### Miscellaneous functions
def vectorized_result(j):
    """
    Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    :param j: index between (0...9)
    :return: 10-dimensational unit vector with a 1.0 in the j'the position and zeros elsewhere.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


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
