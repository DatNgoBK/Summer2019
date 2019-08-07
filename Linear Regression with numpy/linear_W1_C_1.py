import numpy as np
import pandas as pd


data_csv = pd.read_csv('data_linear.csv')
data = data_csv.values
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)

X_add_one = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
print(X_add_one)

class Linear_regression:
	def __init__(self, train_data_X, train_data_Y):
		self.X, self.Y = train_data_X, train_data_Y
		self.W = np.array([0., 0.], dtype=np.float64).reshape(-1, 1)

	def predict(self, X):
		Y_predict = np.array(X).dot(self.W)
		return Y_predict

	def fit(self):
		loop = 0
		while loop < 100000:
			Y_predict = self.predict(self.X)
			d_w0 = np.sum(Y_predict - self.Y)
			d_w1 = np.sum(np.multiply(self.X[:, 1], Y_predict - Y))
			self.W[0, 0] = self.W[0, 0] - 0.0000001 * d_w0
			self.W[1, 0] = self.W[1, 0] - 0.0000001 * d_w1
			loop +=1
		print(self.W)



if __name__ == '__main__':
	model = Linear_regression(X_add_one, Y)
	model.fit()
	Y_predict = model.predict(X_add_one)

		



		