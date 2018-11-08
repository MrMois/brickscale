#!/usr/bin/python3

""" Neural network by Victor Czech, 2018

Todo:

Multiple cost functions implemented
Multiple act functions in single network
Matrix batch training
Momentum

"""




""" IMPORTS

"""

import numpy as np # ext
import os # ext



""" Non linear activation functions

"""

def sigmoid(x, derivative=False):

	if derivative:
		return x * (1 - x)
	else:
		return 1.0 / (1 + np.exp(-x))




""" Cost functions

"""


def quadratic(activation, target, gradient=True):

	if gradient:
		return (activation - target)
	else:
		return 0.5 * (activation - target) ** 2







""" Layer object with basic forward/backward functionality

"""

class Layer(object):



	def __init__(self, input_size, output_size, range=[-1,1], bias=0.75,
		is_output=False, activation=sigmoid, cost=quadratic):

		self.is_output = is_output
		self.activation = activation
		self.cost = cost

		# weight matrix init, with bias added as column

		rng = np.abs(range[0]) + range[1]
		weights = rng * np.random.rand(output_size, input_size) + range[0]
		biases = bias * np.ones((output_size,1))

		self.weights = np.c_[weights, biases]



	def save(self, name, id):

		np.save(name+"/w"+str(id), self.weights)



	def load(self, name, id):

		self.weights = np.load(name+"/w"+str(id)+".npy")



	def forward(self, input):

		self.input = input
		input_b = np.r_[input, [1]] # add bias

		self.out = np.dot(self.weights, input_b)
		self.act = self.activation(self.out)

		return self.out, self.act



	def backward(self, target=None, delta_next=None, w_next=None):

		if self.is_output:

			error = self.cost(self.act, target, gradient=True)
			delta = error * self.activation(self.act, derivative=True) # check it!

		else:

			delta = np.dot(w_next[:,:-1].T, delta_next)
			delta = delta * sigmoid(self.act, derivative=True)

		self.delta_w = np.c_[
				np.dot(
					delta.reshape((len(delta), 1)), 
					self.input.reshape(1,(len(self.input)))),
				delta]

		return delta, self.weights
			


	def apply_delta(self, learning_rate):

		self.weights += -learning_rate * self.delta_w





""" Network object with deep forward/backward functionality

"""

class Net(object):


	def __init__(self, structure=None, cost=quadratic, model=None):

		# init of layers
		self.layers = []

		if model is None:

			self.struct = structure

		else:

			self.struct = np.load(model + "/struct.npy")


		self.size = len(self.struct)

		for l in range(self.size-2):

			input_size = self.struct[l]
			output_size = self.struct[l+1]

			self.layers.append(Layer(input_size, output_size))

		input_size = self.struct[self.size-2]
		output_size = self.struct[self.size-1]

		self.layers.append(Layer(input_size=input_size, 
			output_size=output_size, is_output=True, cost=cost))


		if model is not None:
		
			i = 0 

			for l in self.layers:
				l.load(model,i)
				i += 1
		
		self.cost = cost
			



	def save(self, name="model"):

		if not os.path.exists(name+"/"):
			os.makedirs(name+"/")
			print("Created /" + name + "/")

		np.save(name+"/struct", self.struct)

		i = 0

		for l in self.layers:

			l.save(name, i)
			i += 1

		print("Saved model: " + name)



	def forward(self, input):

		act = np.copy(input)

		for l in self.layers:
			_, act = l.forward(act)

		return act



	def backward(self, target):

		delta, weights = self.layers[-1].backward(target=target)

		for l in reversed(self.layers[:-1]):
			delta, weights = l.backward(delta_next=delta, w_next=weights)



	def apply_delta(self, learning_rate):

		for l in self.layers:
			l.apply_delta(learning_rate)



	def train(self, input, target, learning_rate):

		act = self.forward(input)

		self.backward(target)

		current_cost = self.cost(activation=act, target=target, 
			gradient=False)

		self.apply_delta(learning_rate)

		return act, np.sum(current_cost)





######################### TESTING ONLY ###################################


if __name__ == '__main__':


	print("Running network.py as main")


	# XOR TEST

	import random

	struct = [2,3,1]

	# n = Net(structure=struct)
	n = Net(model="model")

	epochs = 100000
	learning_rate = 0.1

	print_steps = 200
	print_step = 0

	cost = 100

	

	while cost > 0.01:

		input = [random.getrandbits(1), random.getrandbits(1)]

		target = [0]

		if input[0] != input[1]:
			target[0] = 1

		_, cost = n.train(input=input, target=target, 
			learning_rate=learning_rate)

		print(cost)

	n.save()