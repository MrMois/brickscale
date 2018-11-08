#!/usr/bin/python3


from trainingset import Dataset
from network import Net
import cv2
import numpy as np


def scale_colors(x, down=True):
	if down:
		return 1.0*x/256.0
	else:
		return np.int_(x*256)


crop_size = 9
netinput_size = 6
target_size = 6

source = "schwarzwald.jpeg"

data = Dataset(source, crop_size, netinput_size, target_size)


struct = [netinput_size**2,100,91,55,target_size**2]
learning_rate = 0.1
e = 0

model = "modelBrick1"
net = Net(model=model)

average = 0


while True:

	crop, targetdata, inputdata, _, t_img, _ = data.next()

	target = scale_colors(targetdata)
	netinput = scale_colors(inputdata)

	output, cost = net.train(netinput, target, learning_rate)

	average += cost

	if e%1000 == 0:
		print(average)
		average = 0

	if e>=(10**5):

		net.save(model)

		e = 0

		netout_data = scale_colors(output.reshape((6,6)), False)
		cv2.imwrite(model + "/target.jpeg", t_img)
		cv2.imwrite(model + "/netout.jpeg", netout_data)


	e += 1
