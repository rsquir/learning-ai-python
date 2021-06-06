import matplotlib.pyplot as plt
import numpy as np

#each point is length, width, type
data = [[3, 1.5, 1],
		[2, 1, 0],
		[4, 1.5, 1],
		[3, 1, 0],
		[3.5, 0.5, 1],
		[2, 0.5, 0],
		[5.5, 1, 1],
		[1, 1, 0]]
mystery_flower = [4.5, 1]

print(data[1])

#network

#    o      flower type
#   /  \    w1, w2, b
#  o    o   length, width

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()
print (w1, w2, b)

def sigmoid(x):
	return 1/(1 + np.exp(-x))

T = np.linspace(-20, 20, 100)
Y = sigmoid(T)
plt.plot(T, Y)
plt.savefig("giant-nn-12-graph.png")