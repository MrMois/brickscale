#!/usr/bin/python3
# model by Victor Czech, 2018


# imports

from functions import Activation_Function, Cost_Function



class Structure:

	def __init__(self, layers, act_functions, cost_function, 
		weight_range=[-1,1], bias_range=[0.5,0.9]):

		self.layers = layers
		self.size = len(layers)-1

		if isinstance(act_functions, str):
			array = []
			for _ in range(self.size):
				array.append(act_functions)
			self.act_functions = array
		else:
			self.act_functions = act_functions
		
		self.cost_function = cost_function




class Model:


	def __init__(self, structure):
		pass



### ---- Testing ----

if __name__ == '__main__':

	print("Running model.py as main")

	structure = Structure(
		layers=[2,4,3,1],
		act_functions=["sigmoid","sigmoid","sigmoid","sigmoid"],
		cost_function="quadratic"
		)