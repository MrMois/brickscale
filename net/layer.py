#!/usr/bin/python3
# layer by Victor Czech, 2018


import numpy as np

from functions import ActivationFunction


# Layer object with basic forward / backward functionality

class Layer:


    def __init__(self, activation):

        self.activation = ActivationFunction(activation)


    @staticmethod
    def init_with_weights(input_size, output_size, activation,
        w_range, b_range):

        layer = Layer(activation)

        w_multiplier = np.abs(w_range[0] - w_range[1])
        b_multiplier = np.abs(b_range[0] - b_range[1])

        weights = w_multiplier * np.random.rand(output_size, input_size) \
                + w_range[0]

        biases = b_multiplier * np.random.rand(output_size, 1) \
               + b_range[0]

        layer.weights = np.c_[weights, biases]

        return layer

        

    @staticmethod
    def load():
        pass



def main():

    l = Layer.init_with_weights(
            input_size = 3,
            output_size = 2,
            activation = "sigmoid",
            w_range = [-1,1],
            b_range = [0.5,0.9]
        )

    print(l.weights)



if __name__ == '__main__':
    main()