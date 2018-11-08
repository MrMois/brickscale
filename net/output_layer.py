#!/usr/bin/python3
# output layer by Victor Czech, 2018


import numpy as np
from layer import Layer
from functions import CostFunction


class OutputLayer(Layer):


    def __init__(self, cost_func, activation_func):

        self.cost_func = CostFunction(cost_func)

        Layer.__init__(self, activation_func)


    # returns a new output layer with weights
    @staticmethod
    def init_with_weights(cost_func, activation_func,
        input_size, output_size, w_range, b_range):

        layer = OutputLayer(cost_func, activation_func)

        layer.weights = Layer.random_weights(input_size, output_size,
                        w_range,b_range)

        return layer


    # single target backward
    def backward(self, target):

        cost = self.cost_func.compute(self.activation, target)
        error = self.cost_func.gradient(self.activation, target)
        # functions like sigmoid use act instead of out
        self.delta = self.activation_func.gradient(self.activation)
        self.delta *= error

        return self.delta, self.weights, cost



# testing only!
def main():

    layer = OutputLayer.init_with_weights(
            input_size = 3,
            output_size = 2,
            cost_func = "quadratic",
            activation_func = "sigmoid",
            w_range = [-1,1],
            b_range = [0.5,0.9]
        )

    input_vector = np.ones(3)

    target_vector = np.ones(2)

    o, a = layer.forward(input_vector)

    c = layer.cost_func.compute(a, target_vector)

    print(o, a, c)



if __name__ == '__main__':
    main()