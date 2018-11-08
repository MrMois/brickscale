#!/usr/bin/python3


""" training set by Victor Czech, 2018

"""


""" IMPORTS

"""

import cv2 # ext



class Dataset(object):


	def __init__(self, source, crop_size, netinput_size, target_size, 
		gray=True):


		self.source = source
		self.image = cv2.imread(source)

		if gray:
			self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

		self.img_h, self.img_w = self.image.shape[:2]


		self.crop_size = crop_size
		self.netinput_size = netinput_size
		self.target_size = target_size

		self.pos = [0,0]



	def new_pos(self, repeat=True):


		# horizontal first scan
		if self.pos[1] + 1 +  self.crop_size < self.img_w:

			self.pos[1] += 1
			return True

		# vertical second scan
		elif self.pos[0] + 1 + self.crop_size < self.img_h:

			self.pos[1] = 0
			self.pos[0] += 1
			return True

		# reset, image scanned
		elif repeat:

			self.pos = [0,0]
			return True

		else:

			return False



	def next(self, repeat=True):


		if self.new_pos(repeat):

			crop = self.image[self.pos[0]:self.pos[0]+self.crop_size+1,
								self.pos[1]:self.pos[1]+self.crop_size+1]

			center = (int)((self.crop_size - self.target_size)/2)

			target = crop[center:self.target_size+1,
							center:self.target_size+1]

			netinput = cv2.resize(crop, 
				(self.netinput_size, self.netinput_size),
				interpolation=cv2.INTER_CUBIC)

			return crop.ravel(), target.ravel(), netinput.ravel(),\
				crop, target, netinput, 




######################### TESTING ONLY ###################################


if __name__ == '__main__':


	print("Running trainingset.py as main")


	source = "schwarzwald.jpeg"

	crop_size = 90
	netinput_size = 60
	target_size = 60



	dataset = Dataset(source, crop_size, netinput_size, target_size)

	for i in range(20000):

		c, t, n, _, _, _ = dataset.next()

		cv2.imshow("C",c)
		cv2.imshow("T",t)
		cv2.imshow("N",n)

		cv2.waitKey(1)


	cv2.destroyAllWindows()