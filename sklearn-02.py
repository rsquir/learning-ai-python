
from sklearn.linear_model import Perceptron
import numpy as np

X_train = np.array([[255/255, 255/255, 255/255],
					[200/255, 200/255, 200/255],
					[170/255, 190/255, 180/255],
					[100/255, 110/255, 120/255],
					[50/255, 50/255, 50/255],
					[0/255, 0/255, 0/255]])
y_train = np.array([0, 0, 0, 1, 1, 1])
X_test = np.array([[255/255, 255/255, 255/255], [50/255, 50/255, 100/255]])

clf = Perceptron()
clf.fit(X_train, y_train)

print(clf.predict(X_test))
