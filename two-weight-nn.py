# write for
# [0, 0, 0]
# [0, 1, 0]
# [1, 0, 0]
# [1, 1, 1]

#data
#weights & bias

#loop range(500)
#random index
#point

#z = point[0] * w1 + point[1] * w2 + b
#pred = sigmoid(z)

#target = point[2]

#cost = np.square(pred - target)

#dcost = 2 * (pred - target)
#dpred = sigmoid_p(z)

#bias = 1

# some multiplication of derivatives here

#w1 = w1 - learning_rate * dsmthn
#w2 = w2 - learning_rate * dsmthn
#b = b - learning_rate * bias

import matplotlib.pyplot as plt
import numpy as np

data = [[0, 0, 0],	# remove this array
		[0, 1, 0],
		[1, 0, 0],
		[1, 1, 1]]
test = [0, 0] 		#answer is 0

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_p(x):
	return sigmoid(x) * (1 - sigmoid(x))

learning_rate = 0.1

for i in range(500):
	ri = np.random.randint(len(data))
	point = data[ri]

	z = point[0] * w1 + point[1] * w2 + b
	pred = sigmoid(z)

	target = point[2]
	cost = np.square(pred - target)

	dcost_pred = 2 * (pred - target)
	dpred_dz = sigmoid_p(z)

	dz_dw1 = point[0]
	dz_dw2 = point[1]
	dz_db = 1

	dcost_dw1 = dcost_pred * dpred_dz * dz_dw1
	dcost_dw2 = dcost_pred * dpred_dz * dz_dw2
	dcost_db = dcost_pred * dpred_dz * dz_db

	w1 = w1 - learning_rate * dcost_dw1
	w2 = w2 - learning_rate * dcost_dw2
	b = b - learning_rate * dcost_db

z = sigmoid(test[0] * w1 + test[1] * w2 + b)
print('{0:.2f}'.format(z))




