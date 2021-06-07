# AKA "regressor.py" in book
#
# to run:
# python3 pmlc-1-p8.py Chapter01/data_singlevar.txt

import sys
import numpy as np
filename = sys.argv[1]
X = []
y = []
with open(filename, 'r') as f:
	for line in f.readlines():
		xt, yt = [float(i) for i in line.split(',')]
		X.append(xt)
		y.append(yt)

num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])

# Test data
X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])

from sklearn import linear_model

# Create linear regression object
linear_regressor = linear_model.LinearRegression()

# Train the model using the training sets
linear_regressor.fit(X_train, y_train)

import matplotlib.pyplot as plt

y_train_pred = linear_regressor.predict(X_train)
plt.figure()
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')
plt.savefig("graph-plmc-p8-1.png")

# clear graph
plt.clf()

y_test_pred = linear_regressor.predict(X_test)

plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.title('Test data')
plt.savefig("graph-plmc-p8-2.png")

import sklearn.metrics as sm

print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))


import _pickle as pickle

output_model_file = 'saved_model-plmc-1-p8.pk1'

with open(output_model_file, 'wb') as f:
	pickle.dump(linear_regressor, f)

with open(output_model_file, 'rb') as f:
	model_linregr = pickle.load(f)

y_test_pred_new = model_linregr.predict(X_test)
print("New mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))

plt.clf()

plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred_new, color='black', linewidth=4)
plt.title('Test data from file')
plt.savefig("graph-plmc-8-3.png")


