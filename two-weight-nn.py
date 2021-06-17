# write for
# [0, 0, 0]
# [0, 1, 0]
# [1, 0, 0]
# [1, 1, 1]

data
weights & bias

loop range(500)
random index
point

z = point[0] * w1 + point[1] * w2 + b
pred = sigmoid(z)

target = point[2]

cost = np.square(pred - target)

dcost = 2 * (pred - target)
dpred = sigmoid_p(z)

bias = 1

# some multiplication of derivatives here

w1 = w1 - learning_rate * dsmthn
w2 = w2 - learning_rate * dsmthn
b = b - learning_rate * bias

