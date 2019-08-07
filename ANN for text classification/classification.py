import numpy as np 
import tensorflow as tf 

VOB_SIZE = 10110

def read_data(path):
	with open(path, 'r') as f:
		lines = f.read().splitlines()
	Y_train = np.array([int(line.split('<fff>')[0]) for line in lines])
	X_train = []

	for line in lines:
		x = np.zeros(VOB_SIZE)
		features = line.split('<fff>')[2].split()
		for feature in features:
			_id, _val = int(feature.split(':')[0]), float(feature.split(':')[1])
			x[_id] = _val
		X_train.append(x)
	X_train = np.array(X_train)
	return X_train, Y_train
X_train, Y_train = read_data('encode.txt')
X_test, Y_test = read_data('encode1.txt')



class MLP:
	def __init__(self, vob_size, unit_number, learning_rate):
		self.vob_size = vob_size
		self.unit_number = unit_number
		self.learning_rate = learning_rate

		self.X_train = tf.placeholder(dtype=tf.float32, shape=[None, vob_size], name='data_traning_X')
		self.X_test = tf.placeholder(dtype=tf.float32, shape=[None, vob_size], name='data_testing_X')
		self.Y_train = tf.placeholder(dtype=tf.int32, shape=[None], name='data_traning_Y')

	def build_graph(self):
		self.W_1 = tf.get_variable(name='weight_1', shape=(self.vob_size, self.unit_number), initializer=tf.random_normal_initializer(seed=6))
		self.b_1 = tf.get_variable(name='biases_1', shape=(self.unit_number, 1), initializer=tf.random_normal_initializer(seed=6))
		self.W_2 = tf.get_variable(name='weight_2', shape=(self.unit_number, 20), initializer=tf.random_normal_initializer(seed=6))
		self.b_2 = tf.get_variable(name='biases_2', shape=(20, 1), initializer=tf.random_normal_initializer(seed=6))
		linear_hidden = tf.matmul(self.X_train, self.W_1) + tf.transpose(self.b_1)
		hidden = tf.sigmoid(linear_hidden)
		logits = tf.matmul(hidden, self.W_2) + tf.transpose(self.b_2)
		output = tf.nn.softmax(logits)
		y_predict = tf.argmax(output, axis=1)

		labels_one_hot = tf.one_hot(self.Y_train, depth=20)
		loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
		loss = tf.reduce_mean(loss)
		return loss
	def predict(self):
		linear_hidden = tf.matmul(self.X_test, self.W_1) + tf.transpose(self.b_1)
		hidden = tf.nn.relu(linear_hidden)
		logits = tf.matmul(hidden, self.W_2) + tf.transpose(self.b_2)
		output = tf.nn.softmax(logits)
		y_predict = tf.argmax(output, axis=1)
		return y_predict

	def trainner(self, loss):
		train_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
		return train_optimizer

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
			x_train, y_train = self.X_train[self._id_last_batch, :], self.Y_train[self._id_last_batch]
			self.batch = 1
			idx = np.random.permutation(self.X_train.shape[0])
			self.X_train = X_train[idx]
			self.Y_train = Y_train[idx]
			return x_train, y_train
		self.batch += 1
		return X_train[self._id_batch[self.batch - 2], :], Y_train[self._id_batch[self.batch - 2]]



def eval():
	batch_size = 32
	epochs = 200
	model = MLP(VOB_SIZE, 60, 0.01)
	loss = model.build_graph()
	train_op = model.trainner(loss)
	y_pre = model.predict()
	data = Dive_data(X_train, Y_train, batch_size)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		epoch = 0
		while(epoch < epochs):
			epoch = epoch + 1
			i = 0
			loss_r = 0
			while(i < data.batch_num):
				i = i + 1
				x_train, y_train = data.next_batch()
				loss_train, train_o = sess.run([loss, train_op], feed_dict={
					model.X_train : x_train,
					model.Y_train : y_train
					})
				loss_r += loss_train
			y_pre_ = sess.run([y_pre], feed_dict={model.X_test : X_test})
			matches = np.equal(y_pre_, Y_test)
			number_true = np.sum(matches.astype(float))
			print('Epoch: {}/ {}, Loss: {}, Acc-Test: {}'.format(epoch, epochs, loss_r/ i, number_true/ Y_test.shape[0]))



eval()





