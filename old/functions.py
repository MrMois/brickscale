#!/usr/bin/python3
# functions by Victor Czech, 2018


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
			raise Exception("Activation function \"" + str(name) +
			 "\" not available")


	# print available functions

	@staticmethod
	def help(name, function_set):

		options = ""

		for f in function_set:
			options += str(f)

		print("Available " + name + " functions:", options)




# ---- Activation functions

def sigmoid(x, derivative=False):

	if not derivative:
		return 1.0 / (1 + np.exp(-x))
	else:
		return x * (1 - x)



class Activation_Function(Function):


	options = {"sigmoid" : sigmoid}


	def __init__(self, name):
		Function.__init__(self, name, function_set=Activation_Function.options)

	@staticmethod
	def help():
		Function.help("Activation", Activation_Function.options)

	def calculate(self, x):
		return self.f(x=x, derivative=False)


	def derivative(self, y):
		return self.f(x=y, derivative=True)
		



# ---- Cost functions

def quadratic(activation, target, derivative=True):

	if derivative:
		return (activation - target)
	else:
		return 0.5 * (activation - target) ** 2



class Cost_Function(Function):


	options = {"quadratic" : quadratic}
	
	def __init__(self, name):
		Function.__init__(self, name, function_set=Cost_Function.options)

	@staticmethod
	def help():
		Function.help("Cost", Cost_Function.options)

	def calculate(self, activation, target):
		return self.f(activation=activation, target=target, derivative=False)


	def derivative(self, activation, target):
		return self.f(activation=activation, target=target, derivative=True)




### ---- Testing ----

if __name__ == '__main__':

	print("Running functions.py as main")

	a = Activation_Function("sigmoid")
	c = Cost_Function("quadratic")

	print(a.calculate(5))
	print(a.derivative(5))

	print(c.calculate(5,4))
	print(c.derivative(5,4))

	try:
		Activation_Function("Not implemented")
	except Exception as e:
		print(e)
	try:
		Cost_Function("Not implemented")
	except Exception as e:
		print(e)

	Activation_Function.help()
	Cost_Function.help()