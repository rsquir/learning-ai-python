import numpy as np
import matplotlib.pyplot as plt

#interlaced my scaling into the array's
train_data = [[255/255, 255/255, 255/255, 0],  # 1 = white, 2 = black
			  [0/255, 0/255, 0/255, 1],
			  [192/255, 200/255, 180/255, 0],
			  [60/255, 70/255, 65/255, 1]]
test_data = [0/255, 0/255, 0/255] # returns 1
#test_data = [255/255, 255/255, 255/255] # returns 0

w1 = np.random.randn()
w2 = np.random.randn()
w3 = np.random.randn()
b = np.random.randn()

learning_rate = 0.1

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_p(x):
	return sigmoid(x) * (1-sigmoid(x))

for i in range(1000):
	random_idx = np.random.randint(len(train_data))
	point = train_data[random_idx]

	q = point[0] * w1 + point[1] * w2 + point[2] * w3 + b
	pred = sigmoid(q)

	target = point[3]
	cost = np.square(pred - target)

	dcost_pred = 2 * (pred - target)
	dpred_dz =  sigmoid_p(q)

	dz_dw1 = point[0]
	dz_dw2 = point[1]
	dz_dw3 = point[2]
	dz_db = 1

	dcost_dw1 = dcost_pred * dpred_dz * dz_dw1
	dcost_dw2 = dcost_pred * dpred_dz * dz_dw2
	dcost_dw3 = dcost_pred * dpred_dz * dz_dw3
	dcost_db = dcost_pred * dpred_dz * dz_db

	w1 = w1 - learning_rate * dcost_dw1
	w2 = w2 - learning_rate * dcost_dw2
	w3 = w3 - learning_rate * dcost_dw3
	b = b - learning_rate * dz_db

p = sigmoid(test_data[0] * w1 + test_data[1] * w2 + test_data[2] * w3 + b)
#print(q)
print('{0:.2f}'.format(p))




