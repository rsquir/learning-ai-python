import numpy

## Lesson 4
def NN(m1, m2, w1, w2, b):
	z = m1 * w1 + m2 * w2 +b
	return sigmoid(z)

def sigmoid(x):
	return 1/(1 + numpy.exp(-x))

w1 = numpy.random.randn()
w2 = numpy.random.randn()
b = numpy.random.randn()

#print(NN(3, 1.5, w1, w2, b))
#print(NN(2, 1, w1, w2, b))
#print(NN(2, 0.5, w1, w2, b))

## Lesson 7
def cost(b):
	return (b - 4) ** 2

def num_slope(b):
	h = 0.0001
	return (cost (b+h) - cost(b))/h

def slope(b):
	return 2 * (b - 4)

b = 8
for i in range(10):
	b = b - 0.1 * slope (b)
	print(b)