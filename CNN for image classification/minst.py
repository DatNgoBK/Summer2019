import tensorflow as tf 
from keras.datasets import mnist
import numpy as np 

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
gpu_options = tf.GPUOptions(allow_growth=True)
print(X_train.shape)

def conv2d(x, W, b, strides=1):
	conv = tf.nn.conv2d(input=x, filter=W, strides=[1, strides, strides, 1], padding='SAME')
	conv = tf.nn.bias_add(conv, b)
	return tf.nn.relu(conv)

def maxpooling(x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

parameters = {
	'w_1' : tf.get_variable(name='weights_1', shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer(seed=2019)),
	'w_2' : tf.get_variable(name='weights_2', shape=(3, 3, 32, 64), initializer=tf.contrib.layers.xavier_initializer(seed=2019)),
	'w_3' : tf.get_variable(name='weights_3', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer(seed=2019)),
	'w_4' : tf.get_variable(name='weights_4', shape=(7 * 7 * 128, 128), initializer=tf.contrib.layers.xavier_initializer(seed=2019)),
	'w_5' : tf.get_variable(name='weights_5', shape=(128, 10), initializer=tf.contrib.layers.xavier_initializer(seed=2019)),
	'b_1' : tf.get_variable(name='biases_1', shape=(32), initializer=tf.contrib.layers.xavier_initializer(seed=2019)),
	'b_2' : tf.get_variable(name='biases_2', shape=(64), initializer=tf.contrib.layers.xavier_initializer(seed=2019)),
	'b_3' : tf.get_variable(name='biases_3', shape=(128), initializer=tf.contrib.layers.xavier_initializer(seed=2019)),
	'b_4' : tf.get_variable(name='biases_4', shape=(128), initializer=tf.contrib.layers.xavier_initializer(seed=2019)),
	'b_5' : tf.get_variable(name='biases_5', shape=(10), initializer=tf.contrib.layers.xavier_initializer(seed=2019)) 
}



class CNN:
	def __init__(self, learning_rate, img_size):
		self.img_size = img_size
		self.learning_rate = learning_rate
		self.X = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 1], name='X_data')
		self.Y = tf.placeholder(tf.int32, shape=[None], name='Y_data')
		self.X_test = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 1], name='X_data_test')

	def build_graph(self):
		conv1 = conv2d(self.X, parameters['w_1'], parameters['b_1'], strides=1)
		conv1 = maxpooling(conv1, k=2)
		conv2 = conv2d(conv1, parameters['w_2'], parameters['b_2'], strides=1)
		conv2 = maxpooling(conv2, k=2)
		conv3 = conv2d(conv2, parameters['w_3'], parameters['b_3'], strides=1)
		#conv3 = maxpooling(conv3, k=2)
		flatten = tf.reshape(conv3, shape=[-1, 7 * 7 * 128])
		linear = tf.matmul(flatten, parameters['w_4']) + parameters['b_4']
		out_1 = tf.sigmoid(linear)
		logits = tf.matmul(out_1, parameters['w_5']) + parameters['b_5']
		output = tf.nn.softmax(logits)
		y_pre = tf.argmax(output, axis=1)
		labels = tf.one_hot(self.Y, 10)
		loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
		loss = tf.reduce_mean(loss)
		return loss, y_pre

	def trainner(self, loss):
		train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
		return train_op

	def predict(self):
		conv1 = conv2d(self.X_test, parameters['w_1'], parameters['b_1'], strides=1)
		conv1 = maxpooling(conv1, k=2)
		conv2 = conv2d(conv1, parameters['w_2'], parameters['b_2'], strides=1)
		conv2 = maxpooling(conv2, k=2)
		conv3 = conv2d(conv2, parameters['w_3'], parameters['b_3'], strides=1)
		#conv3 = maxpooling(conv3, k=2)
		flatten = tf.reshape(conv3, shape=[-1, 7 * 7 * 128])
		linear = tf.matmul(flatten, parameters['w_4']) + parameters['b_4']
		out_1 = tf.sigmoid(linear)
		logits = tf.matmul(out_1, parameters['w_5']) + parameters['b_5']
		output = tf.nn.softmax(logits)
		y_pre = tf.argmax(output, axis=1)
		return y_pre


class Dive_data:
	def __init__(self, X_train, Y_train, batch_size):
		self.X_train = X_train
		self.Y_train = Y_train
		self.batch_size = batch_size
		self.batch_num = X_train.shape[0] / batch_size
		self._id = np.arange(X_train.shape[0])
		self._id_batch = np.split(self._id[:self._id.shape[0] - self._id.shape[0] % batch_size], self.batch_num)
		self._id_last_batch = self._id[self._id.shape[0] - self._id.shape[0] % batch_size : self._id.shape[0]]
		self.batch = 1


	def next_batch(self):
		if(self.batch == self.batch_num):
			x_train, y_train = self.X_train[self._id_last_batch], self.Y_train[self._id_last_batch]
			self.batch = 1
			idx = np.random.permutation(self.X_train.shape[0])
			self.X_train = X_train[idx]
			self.Y_train = Y_train[idx]
			return x_train, y_train
		self.batch += 1
		return X_train[self._id_batch[self.batch - 2]], Y_train[self._id_batch[self.batch - 2]]


def eval():
	epochs = 100
	batch_size = 128
	model = CNN(0.001, 28)
	loss, y_pre_train = model.build_graph()
	train_op = model.trainner(loss)
	y_pre = model.predict()
	data = Dive_data(X_train, Y_train, batch_size)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		epoch = 0
		best_epoch = 0
		max_true = 0
		l = 0
		while(epoch < epochs):
			epoch = epoch + 1
			i = 0
			loss_r = 0
			number_true_train = 0.
			while(i < data.batch_num):
				i = i + 1
				x_train, y_train = data.next_batch()
				loss_train, y_pre_train_, train_o = sess.run([loss,y_pre_train , train_op], feed_dict={
					model.X : x_train,
					model.Y : y_train
					})
				loss_r += loss_train
				matches_train = np.equal(y_pre_train, y_train)
				number_true_train += np.sum(matches_train.astype(float))
			y_pre_ = sess.run([y_pre], feed_dict={model.X_test : X_test})
			matches = np.equal(y_pre_, Y_test)
			number_true = np.sum(matches.astype(float))
			if number_true > max_true:
				max_true = number_true
				best_epoch = epoch
				l = loss_r / i
			print('Epoch: {}/{}, Loss: {}, Acc-Test: {}, Acc-Train: {}'.format(epoch, epochs, loss_r/ i, number_true/ Y_test.shape[0], number_true_train*1./Y_train.shape[0]))

		print('Best Acc-Test:{} at epoch:{} with loss:{}'.format(max_true/ Y_test.shape[0] * 100, best_epoch, l))


eval()

