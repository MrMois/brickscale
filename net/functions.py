#!/usr/bin/python3
# functions by Victor Czech, 2018

"""
Wrapper for activation and cost functions.
Functions can be accessed via compute(x) and gradient(y), where number of
parameters may vary. List of available functions in specific class
constructor, see "function_set".
Some functions, e.g. "sigmoid", take their output as parameter for faster
gradient computing.
"""


import numpy as np


class Function:

    def __init__(self, name, function_set):

        self.name = name
        self.function = function_set[name]


class ActivationFunction(Function):

    @staticmethod
    def sigmoid(x, gradient):

        if not gradient:
            return 1. / (1. + np.exp(-x))
        else:
            # x has to be sigmoid(x) !
            return x*(1 - x)

    def __init__(self, name):

        # list of available functions
        function_set = {
            "sigmoid": ActivationFunction.sigmoid
        }

        Function.__init__(self, name, function_set)

    def gradient(self, y):

        return self.function(y, gradient=True)

    def compute(self, x):

        return self.function(x, gradient=False)


class CostFunction:

    @staticmethod
    def quadratic(activation, target, gradient):

        if not gradient:
            return 0.5 * (activation - target) ** 2
        else:
            return (activation - target)

    def __init__(self, name):

        # list of available functions
        function_set = {
            "quadratic": CostFunction.quadratic
        }

        Function.__init__(self, name, function_set)

    def gradient(self, activation, target):

        return self.function(activation, target, gradient=True)

    def compute(self, activation, target):

        return self.function(activation, target, gradient=False)


def main():

    a = ActivationFunction("sigmoid")

    x = 5

    act = a.compute(x)
    d_act = a.gradient(act)  # gradient of x

    print(x, act, d_act)

    c = CostFunction("quadratic")

    target = 3

    loss = c.compute(act, target)
    d_loss = c.gradient(act, target)

    print(loss, d_loss)


if __name__ == "__main__":
    main()
