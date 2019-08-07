import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_csv('dataset.csv').values

X = data[:, 0:2].reshape(-1, 2)
Y = data[:, 2].reshape(-1, 1)

X_add_one = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

def sigmod(x):
	return 1 / (1 + np.exp(-x))

class Logistic_regression:
	def __init__(self, train_data_X, train_data_Y):
		self.X = train_data_X
		self.Y = train_data_Y
		self.W = np.array([0., 1., 2.]).reshape(-1, 1)

	def predict(self, X):
		Y_predict = sigmod((X.dot(self.W)).T)
		return Y_predict.reshape(-1, 1)

	def fit(self):
		loop = 0
		while loop < 1000000:
			Y_predict = self.predict(self.X)
			self.W[0, 0] -= 0.0001 * np.sum(Y_predict - self.Y)
			self.W[1, 0] -= 0.0001 * np.sum(np.multiply(self.X[:, 1], Y_predict - self.Y))
			self.W[2, 0] -= 0.0001 * np.sum(np.multiply(self.X[:, 2], Y_predict - self.Y))
			loop += 1

		print(self.W)



if __name__ == '__main__':
	model = Logistic_regression(X_add_one, Y)
	model.fit()
