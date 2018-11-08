# network version 2 by Victor Czech, 2018

# imports

import pickle
import numpy as np
from layer import Layer, OutputLayer


# Network object with deep forward/backward functionality

class Network:


	def __init__(
			self,
			model,
			struct,
			activation="sigmoid",
			cost="quadratic",
			load_weights=False,
			weight_range=[-1,1],
			bias_range=[0.5,0.95]
		):

		self.model = model
		self.struct =struct
		self.activation = activation
		self.cost = cost
		
		# size is the number of weight matrices
		self.size = len(self.struct) - 1

		# init of layers

		self.layers = []


		for l in range(self.size):

			if l == self.size - 1:
				self.layers.append(
					OutputLayer(
						model=self.model,
						ID=l,
						activation=self.activation,
						load_weights=load_weights,
						in_size=self.struct[l],
						out_size=self.struct[l+1],
						weight_range=weight_range,
						bias_range=bias_range,
						cost=self.cost
					)
				)

			else:
				self.layers.append(
					Layer(
						model=self.model,
						ID=l,
						activation=self.activation,
						load_weights=load_weights,
						in_size=self.struct[l],
						out_size=self.struct[l+1],
						weight_range=weight_range,
						bias_range=bias_range
					)
				)



	def save(self):

		# create folder
		if not os.path.exists(self.model+"/"):
			os.makedirs(self.model+"/")
			print("Created /" + self.model + "/")

		for l in self.layers:
			l.save()


		cfg = {
			"model": self.model,
			"struct": self.struct,
			"activation": self.activation,
			"cost": self.cost
		}

		with open(self.model + "/cfg.pkl", "wb+") as file:
			pickle.dump(cfg, file, pickle.HIGHEST_PROTOCOL)

		print("Saved model: " + self.model)



	@staticmethod
	def load(model):

		with open(self.model + "/cfg.pkl", "rb") as file:
			cfg = pickle.load(file)

		return Network(
			model=cfg["model"],
			struct=cfg["struct"],
			activation=cfg["activation"],
			cost=cfg["activation"]
		)



	def forward(self, input):

		act = np.copy(input)

		for l in self.layers:
			_, act = l.forward(act)

		return act



	def backward(self, target):

		delta, weights, current_cost = self.layers[-1].backward(target=target)

		for l in reversed(self.layers[:-1]):
			delta, weights = l.backward(delta_next=delta, w_next=weights)

		return current_cost



	def apply_delta(self, learning_rate):

		for l in self.layers:
			l.apply_delta(learning_rate)



	def train(self, input, target, learning_rate):

		act = self.forward(input)

		current_cost = self.backward(target)

		self.apply_delta(learning_rate)

		return act, current_cost



### ---- Testing ----

if __name__ == '__main__':

	print("Running network.py as main")

	# XOR TEST

	import random

	n = Network(
		model="ModelXOR",
		struct=[2,4,3,1],
		activation="sigmoid",
		cost="quadratic"
	)
	
	for l in n.layers:
		print(l.weights)
	
	epochs = 2
	learning_rate = 0.1

	print_steps = 200
	print_step = 0

	cost_sum = 100

	for _ in range(epochs):

		input = [random.getrandbits(1), random.getrandbits(1)]

		target = [0]

		if input[0] != input[1]:
			target[0] = 1

		act, cost = n.train(input=input, target=target, 
			learning_rate=learning_rate)

		cost_sum = sum(cost)

		print(input, target, act)