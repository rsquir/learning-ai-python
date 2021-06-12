import numpy as np
import matplotlib.pyplot as plt

train_data = [[255, 255, 255, 1],  # 1 = white, 2 = black
			  [0, 0, 0, 0]]
test_data = [255, 255, 255]

w1 = np.random.randn()
w2 = np.random.randn()
w3 = np.random.randn()
b = np.random.randn()

learning_rate = 0.2

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

for i in range(100):
	random_idx = np.random.randint(len(train_data))
	point = train_data[random_idx]

	q = point[0] * w1 + point[1] * w2 + point[2] * w3 + b
	pred = sigmoid(q)

	target = point[3]
	cost = np.square(target - pred)

	d_pred = sigmoid(q) * (1 - sigmoid(q))
	d_cost = 2 * (target - pred)

	dz_dw1 = point[0]
	dz_dw2 = point[1]
	dz_dw3 = point[2]
	dz_db = 1

	dcost_dw1 = d_cost * d_pred * dz_dw1
	dcost_dw2 = d_cost * d_pred * dz_dw2
	dcost_dw3 = d_cost * d_pred * dz_dw3
	dcost_db = d_cost * d_pred * dz_db

	w1 = w1 - learning_rate * dcost_dw1
	w2 = w2 - learning_rate * dcost_dw2
	w3 = w3 - learning_rate * dcost_dw3
	b = b - learning_rate * dz_db

p = sigmoid(test_data[0] * w1 + test_data[1] * w2 + test_data[2] * w3 + b)

print(p)




