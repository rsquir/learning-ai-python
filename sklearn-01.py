from sklearn.linear_model import Perceptron
import numpy as np

X_train = np.array([[3, 1.5],
					[2, 1],
					[4, 1.5],
					[3, 1],
					[3.5, 0.5],
					[2, 0.5],
					[5.5, 1],
					[1, 1]])
y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0])
X_test = np.array([[4.5, 1], [0, 0]]) # should be 1 and 0

clf = Perceptron()
clf.fit(X_train, y_train)

print(clf.predict(X_test))