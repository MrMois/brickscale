# functions version 2 by Victor Czech, 2018



# imports

import numpy as np


# Function super class, usage with subclasses

class Function(object):

	def __init__(self, name, function_set):

		self.name = name
		self.function_set = function_set

		if name in function_set:
			self.f = function_set[name]
		else:
			msg = "Function \"" + str(name) + "\" not available"
			raise Exception(msg)


class Activation_Function(Function):

	
	# available functions as methods

	def sigmoid(x, derivative=False):

		if not derivative:
			return 1.0 / (1 + np.exp(-x))
		else:
			return x * (1 - x)

	# register of implemented functions
	
	function_set = {
		"sigmoid" : sigmoid
	}


	def __init__(self, name):
		Function.__init__(
			self, 
			name=name, 
			function_set=Activation_Function.function_set
		)


	# evaluation
		
	def evaluate(self, x):
		return self.f(x=x, derivative=False)

	
	# gradient
	
	def gradient(self, y):
		return self.f(x=y, derivative=True)

		

class Cost_Function(Function):

	# available functions as methods

	def quadratic(activation, target, derivative=True):

		if derivative:
			return (activation - target)
		else:
			return 0.5 * (activation - target) ** 2
	
	# register of implemented functions

	function_set = {
		"quadratic" : quadratic
	}

	
	def __init__(self, name):
		Function.__init__(
			self, 
			name=name, 
			function_set=Cost_Function.function_set
		)

		
	# evaluation
	
	def evaluate(self, activation, target):
		return self.f(
			activation=activation,
			target=target,
			derivative=False
		)

		
	# gradient
	
	def gradient(self, activation, target):
		return self.f(
			activation=activation,
			target=target,
			derivative=True
		)



### ---- Testing ----

if __name__ == '__main__':

	print("Running functions.py as main")

	a = Activation_Function("sigmoid")
	c = Cost_Function("quadratic")

	print(a.evaluate(5))
	print(a.gradient(5))

	print(c.evaluate(5,4))
	print(c.gradient(5,4))

	try:
		Activation_Function("Not implemented")
	except Exception as e:
		print(e)
	try:
		Cost_Function("Not implemented")
	except Exception as e:
		print(e)
