#!/usr/bin/python3
# network by Victor Czech, 2018


import numpy as np
from layer import Layer
from output_layer import OutputLayer


class Network:

    def __init__(self):

        self.layers = []

    def new_network(cost_func, activation_func, struct, w_range, b_range):

        net = Network()

        net.cost_func = cost_func
        net.activation_func = activation_func
        net.struct = struct

        # stops
        for input_size, output_size in zip(struct, struct[1:-1]):
            layer = Layer.new_layer(
                        activation_func, input_size,
                        output_size, w_range, b_range
                        )
            net.layers.append(layer)

        output_layer = OutputLayer.new_output_layer(
                            cost_func, activation_func, struct[-2],
                            struct[-1], w_range, b_range
                            )

        net.layers.append(output_layer)

        return net

    @staticmethod
    def load():
        pass

    # deep forward
    def forward(self, input):

        activation = np.copy(input)

        for l in self.layers:
            _, activation = l.forward(activation)

        return activation

    # deep backward
    def backward(self, target):

        delta, weights, current_cost = self.layers[-1].backward(target)

        for l in reversed(self.layers[:-1]):
            delta, weights = l.backward(delta_next=delta, w_next=weights)

        return current_cost

    def apply_delta(self, learning_rate):

        for l in self.layers:
            l.apply_delta(learning_rate)

    def train(self, input, target, learning_rate):

        activation = self.forward(input)
        current_cost = self.backward(target)

        self.apply_delta(learning_rate)

        return activation, current_cost


def main():

    import random

    n = Network.new_network(
        cost_func="quadratic",
        activation_func="sigmoid",
        struct=[2, 4, 3, 1],
        w_range=[-1, 1],
        b_range=[0.5, 0.9]
        )

    for l in n.layers:
        print(l.weights)

    epochs = 1000
    learning_rate = 0.1

    for _ in range(epochs):

        input = [random.getrandbits(1), random.getrandbits(1)]

        target = [0]

        if input[0] != input[1]:
            target[0] = 1

        act, cost = n.train(
                        input=input, target=target,
                        learning_rate=learning_rate
                        )

        # cost_sum = sum(cost)

        print(input, target, act)

if __name__ == "__main__":
    main()
