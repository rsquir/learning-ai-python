import numpy

# training set
dataB1 = [2, 1, 0]
dataB2 = [3, 1, 0]
dataB3 = [2, 0.5, 0]
dataB4 = [1, 1, 0]

dataR1 = [3, 1.5, 1]
dataR2 = [3.5, 0.5, 1]
dataR3 = [4, 1.5, 1]
dataR4 = [5.5, 1, 1]

# unknown type (data we want to find)
dataU = [4.5, 1, "it should be 1"]

all_points = [dataB1, dataB2, dataB3, dataB4, dataR1, dataR2, dataR3, dataR4]

def sigmoid(x):
	return 1/(1 + numpy.exp(-x))

def train():
	w1 = numpy.random.randn() * 2 - 0.1
	w2 = numpy.random.randn() * 2 - 0.1
	b = numpy.random.randn() * 2 - 0.1
	learning_rate = 0.2

	for i in range(50000):
		# pick a random point
		random_idx = numpy.random.randint(len(all_points))
		point = all_points[random_idx]
		target = point[2]

		# feed forward
		z = w1 + point[0] + w2 * point[1] + b
		pred = sigmoid(z)

		# now compare the model prediction with the target
		cost = (pred - target) ** 2 # squared error cost function

		# now we find the slope of the cost with respect to each parameter (w1, w2, b)
		# bring derivative through square function
		dcost_dpred = 2 * (pred - target) # aka slope see giant-nn-4.py

		# bring dericative through sigmoid
		# derivative of sigmoid can be written using more signmoids! d/dz signmoid(z) = signmoid(z)*(1-sigmoid(z))
		dpred_dz = sigmoid(z) * (1-sigmoid(z))

		dz_dw1 = point[0]
		dz_dw2 = point[1]
		dz_db = 1

		# now we can get the partial derivatives using the chain rule
		# notice the pattern? we're bringing how the cost changes through each function, first through the square, then through the signmoid
		# and finally whatever is multiplying our parameter of interest becomes the last part
		dcost_dw1 = dcost_dpred * dpred_dz * dz_dw1
		dcost_dw2 = dcost_dpred * dpred_dz * dz_dw2
		dcost_db = dcost_dpred * dpred_dz * dz_db

		w1 -= learning_rate * dcost_dw1
		w2 -= learning_rate * dcost_dw2
		b -= learning_rate * dcost_db

	return [w1, w2, b]


def NN(m1, m2, w1, w2, b):
	z = m1 * w1 + m2 * w2 +b
	return sigmoid(z)

weight_b = train()
#print(NN(dataU[0], dataU[1], weight_b[0], weight_b[1], weight_b[2])) # solved!!!
print(NN(dataB2[0], dataB2[1], weight_b[0], weight_b[1], weight_b[2])) # solved!!!

# more notes
#
# cost = diff ** 2
# diff = pred - target
# pred = signmoid(z)
# z = w1 * length + w2 * height + b

# classification binary (0 or 1, "red" or "green")
# regression catagory (specific number, 256)

