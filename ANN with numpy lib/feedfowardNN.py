import numpy as np
import pandas as pd
from scipy.special import expit
np.random.seed(2018)

def sigmod(x):
	return 1 / (1 + np.exp(-x))

class NeuralNetwork:
	def __init__(self, layers, learning_rate, training_data_X, training_data_Y):
		self.layers = layers
		self.learning_rate = learning_rate
		self.training_data_X = training_data_X
		self.training_data_Y = training_data_Y

		#Khoi tao tham so W, b
		self.W = [0]
		self.b = [0]

		for i in range(len(layers) - 1):
			W_temp = np.random.randn(layers[i], layers[i+1]) / layers[i]
			b_temp = np.zeros((layers[i+1], 1)) 
			self.W.append(W_temp)
			self.b.append(b_temp)

	def predict(self, X):
		self.A = []
		self.A.append(X)
		for i in range(len(self.layers) - 1):
			Z_temp = self.A[i].dot(self.W[i + 1]) + (self.b[i+1].T)
			A_temp = sigmod(Z_temp)
			self.A.append(A_temp)
		return self.A[-1]

	def fit_model(self):
		k = len(self.layers) - 1
		for _ in range(100000):
			d_A = []
			d_W = []
			d_b = []
			Y_predict = self.predict(self.training_data_X)
			d_A_temp = - (self.training_data_Y / Y_predict - (1 - self.training_data_Y)/ (1 - Y_predict))
			d_A.append(d_A_temp)
			for i in range(len(self.layers) - 1):				
				temp = d_A[-1] * self.A[k - i] * (1 - self.A[k - i])
				d_W_temp = self.A[k - i - 1].T.dot(temp)
				d_b_temp = (np.sum(temp, 0)).reshape(-1, 1)
				d_A_temp = (temp).dot(self.W[k - i].T)
				d_W.append(d_W_temp)
				d_b.append(d_b_temp)
				d_A.append(d_A_temp)
			for i in range(len(self.layers) - 1):
				self.W[k - i] = self.W[k - i] - self.learning_rate * d_W[i]
				self.b[k - i] = self.b[k - i] - self.learning_rate * d_b[i]
				

if __name__ == '__main__':
	data_csv = pd.read_csv('dataset.csv')
	data = data_csv.values
	training_data_X = data[:, 0:2]
	training_data_Y = data[:, 2].reshape(-1, 1)
	model = NeuralNetwork([2, 2, 1], 0.1, training_data_X, training_data_Y)
	model.fit_model()
	print(model.W)
	print(model.b)

	



