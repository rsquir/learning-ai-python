# AKA housing.py

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

housing_data = datasets.load_boston()

X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(X_train, y_train)

ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
ab_regressor.fit(X_train, y_train)

y_pred_dt = dt_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_dt)
evs = explained_variance_score(y_test, y_pred_dt)
print("####Decision Tree Performance####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

y_pred_ab = ab_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_ab)
evs = explained_variance_score(y_test, y_pred_ab)
print("#### AdaBoos performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

def plot_feature_importance(feature_importances, title, feature_names, graph_num):
	# Normalize importance values
	feature_importances = 100.0 * (feature_importances / max(feature_importances))
	
	# Sort the index values and flip them so that they are arranced in decreasing order of importance
	index_sorted = np.flipud(np.argsort(feature_importances))

	# Center the location of the labels on the X-axis (for display purposes only)
	#pos = np.arange(index_sorted.shape[0]) + 0.5
	# couldn't get arrange to work but other then that code's fine

	# Plot the bar graph
	plt.figure()
	plt.bar(pos, feature_names[index_sorted], align='center')
	plt.xticks(pos, feature_names[index_sorted])
	plt.ylabel('Relative Importance')
	plt.title(title)
	plt.savefig("bar-pmlc-1-p18-" + str(graph_num) + ".png")

plot_feature_importance(dt_regressor.feature_importances_, 'Decision Tree regressor', housing_data.feature_names, 1)
plot_feature_importance(ab_regressor.feature_importances_, 'Adaboost regressor', housing_data.feature_names, 2)


