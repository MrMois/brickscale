#!/usr/bin/python3
# layer by Victor Czech, 2018


# imports

import pickle
import numpy as np
from functions import Activation_Function, Cost_Function



"""

Example layer cfg:

cfg = {
	"model": "TestModel",
	"ID": 0,
	"activation": "sigmoid",
	"load_weights": False,
	"shape": [3,2]
}

"""


# Layer object with basic forward/backward functionality

class Layer:

	def __init__(self, cfg):

		self.model = cfg["model"]
		self.ID = cfg["ID"]
		self.activation = Activation_Function(cfg["activation"])

		# config later used for saving, empty because only updated on save
		self.cfg = {}
		
		if cfg["load_weights"]:
			self.weights = np.load(self.model+"/l"+str(self.ID)+"weights.npy")

		else:
			self.weights = self.create_weights(cfg["shape"])


	def create_weights(self, shape):

		in_size, out_size = shape[0], shape[1]
		weight_range, bias_range = [-1,1], [0.5,0.95]

		# weight matrix init, with bias added as column

		w_multi = np.abs(weight_range[0] - weight_range[1])
		b_multi = np.abs(bias_range[0] - bias_range[1])
		weights = w_multi * np.random.rand(out_size, in_size) + weight_range[0]

		biases = b_multi * np.random.rand(out_size,1) + bias_range[0]

		return np.c_[weights, biases]


	def save(self):

		self.cfg["model"] = self.model
		self.cfg["ID"] = self.ID
		self.cfg["activation"] = self.activation.name

		path = self.model + "/l" + str(self.ID)

		with open(path + "cfg.pkl", "wb+") as file:
			pickle.dump(self.cfg, file, pickle.HIGHEST_PROTOCOL)

		np.save(path + "weights.npy", self.weights)



	def forward(self, input):

		self.input = input
		input_b = np.r_[input, [1]] # add bias

		self.out = np.dot(self.weights, input_b)
		self.act = self.activation.calculate(self.out)

		return self.out, self.act


	def backward(self, w_next, delta_next):

		delta = np.dot(w_next[:,:-1].T, delta_next)
		self.delta = delta * self.activation.derivative(self.act)

		return self.delta, self.weights

			
	def apply_delta(self, learning_rate):

		self.delta_w = np.c_[
				np.dot(
					self.delta.reshape((len(self.delta), 1)), 
					self.input.reshape(1,(len(self.input)))),
				self.delta]

		self.weights += -learning_rate * self.delta_w
	
	

class OutputLayer(Layer):

	def __init__(self, cfg):

		Layer.__init__(self, cfg)

		self.cost = Cost_Function(cfg["cost"])


	def save(self):

		self.cfg["cost"] = self.cost.name
		Layer.save(self)


	def backward(self, target):


		cost = self.cost.calculate(self.act, target)
		error = self.cost.derivative(self.act, target)
		self.delta = error * self.activation.derivative(self.act)

		return self.delta, self.weights, cost

		



### ---- Testing ----

if __name__ == '__main__':

	print("Running layer.py as main")

	cfg = {
		"model": "TestModel",
		"ID": 0,
		"activation": "sigmoid",
		"load_weights": False,
		"shape": [3,2]
	}

	l = Layer(cfg)

	print(l.weights)

	l.save()

	cfg["load_weights"] = True

	l = Layer(cfg)

	print(l.weights)