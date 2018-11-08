# layer version 2 by Victor Czech, 2018

# imports

import numpy as np
from functions import Activation_Function, Cost_Function


""" Config for a layer

example_cfg = {
	"model" : "TestmodelName", 	# folder/name of model
	"ID" : 3, 					# position in network
	"activation" : "sigmoid",
	"load_weights" : False,		# if false new ones generated
	"in_size" : 3,
	"out_size" : 2,
	"weight_range" : [-1,1],	# optional, defaults listed here
	"bias_range" : [0.5,0.95],	# optional, defaults listed here
	"cost" : "quadratic"		# only used if type OutputLayer
}

"""


# Layer object with basic forward/backward functionality

class Layer:

	# example cfg see above

	def __init__(
			self,
			model,
			ID,
			activation="sigmoid",
			load_weights=False,
			in_size=None,
			out_size=None,
			weight_range=[-1,1],
			bias_range=[0.5,0.95]
		):

		self.model = model
		self.ID = ID
		self.activation = Activation_Function(activation)
		
		if load_weights:
			self.weights = np.load(self.model+"/l"+str(self.ID)+"weights.npy")

		else:
			self.weights = self.create_weights(
				in_size = in_size,
				out_size = out_size,
				 # have default values in header
				weight_range = weight_range,
				bias_range = bias_range
			)


	def create_weights(self, in_size, out_size, weight_range, bias_range):

		# weight matrix init, with bias added as column

		w_multi = np.abs(weight_range[0] - weight_range[1])
		b_multi = np.abs(bias_range[0] - bias_range[1])
		weights = w_multi * np.random.rand(out_size, in_size) + weight_range[0]

		biases = b_multi * np.random.rand(out_size,1) + bias_range[0]

		return np.c_[weights, biases]


	def save(self):
	
		# weights saved in "MODEL"/l"LAYERID"weights.npy

		path = self.model + "/l" + str(self.ID)
		np.save(path + "weights.npy", self.weights)


	def forward(self, input):

		self.input = input
		input_b = np.r_[input, [1]] # add bias

		self.out = np.dot(self.weights, input_b)
		self.act = self.activation.evaluate(self.out)

		return self.out, self.act


	def backward(self, w_next, delta_next):

		delta = np.dot(w_next[:,:-1].T, delta_next)
		# functions like sigmoid use act instead of out
		self.delta = delta * self.activation.gradient(self.act)

		return self.delta, self.weights

			
	def apply_delta(self, learning_rate):

		self.delta_w = np.c_[
				np.dot(
					self.delta.reshape((len(self.delta), 1)), 
					self.input.reshape(1,(len(self.input)))),
				self.delta]

		self.weights += -learning_rate * self.delta_w
	
	

class OutputLayer(Layer):

	def __init__(
			self,
			model,
			ID,
			activation="sigmoid",
			load_weights=False,
			in_size=None,
			out_size=None,
			weight_range=[-1,1],
			bias_range=[0.5,0.95],
			cost="quadratic"
		):

		Layer.__init__(
			self=self,
			model=model,
			ID=ID,
			activation=activation,
			load_weights=load_weights,
			in_size=in_size,
			out_size=out_size,
			weight_range=weight_range,
			bias_range=bias_range,
		)

		self.cost = Cost_Function(cost)


	def backward(self, target):


		cost = self.cost.evaluate(self.act, target)
		error = self.cost.gradient(self.act, target)
		# functions like sigmoid use act instead of out
		self.delta = error * self.activation.gradient(self.act)

		return self.delta, self.weights, cost

		



### ---- Testing ----

if __name__ == '__main__':

	print("Running layer.py as main")
	

	l = Layer(
		model = "TestmodelName", 	# folder/name of model
		ID = 3, 					# position in network
		activation = "sigmoid",
		load_weights = False,		# if false new ones generated
		in_size = 3,
		out_size = 2,
		weight_range = [-1,1],	# optional, defaults listed here
		bias_range = [0.5,0.95]	# optional, defaults listed here
	)
	print(l.weights)

	l.save()

	l = Layer(
		model = "TestmodelName", 	# folder/name of model
		ID = 3, 					# position in network
		activation = "sigmoid",
		load_weights = True,		# if false new ones generated
		in_size = 3,
		out_size = 2,
		weight_range = [-1,1],	# optional, defaults listed here
		bias_range = [0.5,0.95]	# optional, defaults listed here
	)
	print(l.weights)
	
	l = OutputLayer(
		model = "TestmodelName", 	# folder/name of model
		ID = 3, 					# position in network
		activation = "sigmoid",
		load_weights = True,		# if false new ones generated
		in_size = 3,
		out_size = 2,
		weight_range = [-1,1],	# optional, defaults listed here
		bias_range = [0.5,0.95],	# optional, defaults listed here
		cost = "quadratic"
	)
	print(l.weights, l.cost.name)
	