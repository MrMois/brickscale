#!/usr/bin/python3
# layer by Victor Czech, 2018

"""
Todo:
- Add momentum

"""


import numpy as np
from functions import ActivationFunction


# Layer object with basic forward / backward functionality

class Layer:

    def __init__(self, activation_func):

        self.activation_func = ActivationFunction(activation_func)

    # returns a new layer with weights
    @staticmethod
    def init_with_weights(
            activation_func, input_size,
            output_size, w_range, b_range
            ):

        layer = Layer(activation_func)

        layer.weights = Layer.random_weights(
                                input_size, output_size,
                                w_range, b_range)

        return layer

    # returns a matrix with weights and bias
    @staticmethod
    def random_weights(input_size, output_size, w_range, b_range):

        w_multiplier = np.abs(w_range[0] - w_range[1])
        b_multiplier = np.abs(b_range[0] - b_range[1])

        weights = w_multiplier * np.random.rand(output_size, input_size) \
            + w_range[0]

        # extra row for bias
        biases = b_multiplier * np.random.rand(output_size, 1) + b_range[0]

        # append bias vector to weight matrix
        return np.c_[weights, biases]

    # TODO:
    def save():
        pass

    # TODO:
    @staticmethod
    def load():
        pass

    # single input forward
    def forward(self, input):

        self.input = input
        input_biased = np.r_[input, [1]]  # add bias

        self.output = np.dot(self.weights, input_biased)
        self.activation = self.activation_func.compute(self.output)

        return self.output, self.activation

    # single target backward
    def backward(self, w_next, delta_next):

        delta = np.dot(w_next[:, :-1].T, delta_next)
        # functions like sigmoid use activatiob instead of output
        self.delta = self.activation_func.gradient(self.activation)
        self.delta *= delta

        # next layer needs weights as w_next
        return self.delta, self.weights

    # correct weights, without momentum
    def apply_delta(self, learning_rate):

        self.delta_w = np.c_[
                np.dot(
                    self.delta.reshape((len(self.delta), 1)),
                    self.input.reshape(1, (len(self.input)))),
                self.delta]

        self.weights += -learning_rate * self.delta_w


# testing only!
def main():

    layer = Layer.init_with_weights(
            input_size=3,
            output_size=2,
            activation_func="sigmoid",
            w_range=[-1, 1],
            b_range=[0.5, 0.9]
        )

    input_vector = np.ones(3)

    o, a = layer.forward(input_vector)

    print(o, a)


if __name__ == '__main__':
    main()
