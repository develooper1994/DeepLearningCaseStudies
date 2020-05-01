import numpy as np


def sigmoid(x, deriv=False):
    f = 1 / (1 + np.exp(-x))
    df = f*(1-f)
    if deriv:
        return df # np.exp(-x) / (1 + np.exp(-x)) ** 2
    return 1 / (1 + np.exp(-x))


np.random.seed(1)
weights = np.random.randn(1, 3)

training = np.array([
    [np.array([0, 0, 0]).reshape(1, -1), 1],
    [np.array([0, 0, 1]).reshape(1, -1), 0],
    [np.array([0, 1, 0]).reshape(1, -1), 0],
    [np.array([0, 1, 1]).reshape(1, -1), 0],
    [np.array([1, 0, 0]).reshape(1, -1), 1],
    [np.array([1, 0, 1]).reshape(1, -1), 0],
    [np.array([1, 1, 0]).reshape(1, -1), 0],
    [np.array([1, 1, 1]).reshape(1, -1), 1],
])

for iter in range(training.shape[0]):
    # forward propagation
    a_layer1 = training[iter][0]
    z_layer2 = np.dot(weights, a_layer1.reshape(-1, 1))
    a_layer2 = sigmoid(z_layer2)
    hypothesis_theta = a_layer2

    # back propagation
    delta_neuron1_layer2 = (a_layer2 - training[iter][1]) * sigmoid(a_layer2, deriv=True)
    Delta_neuron1_layer2 = np.dot(delta_neuron1_layer2, a_layer1)
    update = Delta_neuron1_layer2
    weights = weights - update

# test the network
x = np.array([0, 0, 1])
print(sigmoid(np.dot(weights, x.reshape(-1, 1))))

x = np.array([0, 1, 1])
print(sigmoid(np.dot(weights, x.reshape(-1, 1))))

x = np.array([1, 1, 1])
print(sigmoid(np.dot(weights, x.reshape(-1, 1))))
