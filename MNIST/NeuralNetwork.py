import numpy as np


def create_weights(prev_layer, next_layer):
    # return np.random.randn(next_layer, prev_layer)*np.sqrt(2 / 300)
    return np.random.rand(next_layer, prev_layer)


def create_biases(no_layer):
    return np.reshape(np.random.randint(2, size=no_layer), (no_layer, 1))


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def diff_sigmoid(x):
    return (1/(1 + np.exp(-x)))*(1 - 1/(1 + np.exp(-x)))


def mse(received, expected):
    return np.mean((received - expected) ** 2)


def diff_mse(received, expected):
    return 2*(received - expected)/received.size


class Layer:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

        self.weights = create_weights(self.outputs, self.inputs)
        self.biases = create_biases(self.outputs)

    def study(self, current_state):
        new_state = sigmoid(np.dot(self.weights, current_state) + self.biases)
        return new_state


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def study(self, in_data):
        next_state = in_data
        for layer in self.layers:
            next_state = layer.study(next_state)
        return next_state

    # MSE here maybe or outside as you wish
    # def loss(self, received, expected):
    #     return sigmoid(received, expected)
