import tensorflow as tf 
import numpy as np 

with open('glove.840B.300d.txt', 'r') as f:
	items = f.read().splitlines()
dict_glove = dict([(item.split()[0], np.array(item.split()[1:], np.float32)) for item in items])

with open('vob_w2v.txt', 'r') as f:
	lines = f.read().splitlines()
total_words = [line.split(':')[0] for line in lines]
pre_train_model = []
for word in total_words:
	pre_train_model.append(dict_glove[word])
pre_train_model = np.array(pre_train_model)
print(pre_train_model.shape)



VOB_SIZE = len(lines)
EMBEDDING_SIZE = 300
MAX_DOC_LENGTH = 500
NUM_CLASS = 20

with open('encode_w2v.txt', 'r') as f:
	data = f.read().splitlines()
Y_train = []
X_train = []
sentence_lengths = []
for line in data:
	label, name, doc_len, text = line.split('<fff>')
	Y_train.append(int(label))
	X_train.append(text.split()[:MAX_DOC_LENGTH])
	sentence_lengths.append(int(doc_len))
Y_train = np.array(Y_train)
X_train = np.array(X_train)
sentence_lengths = np.array(sentence_lengths)

with open('encode_w2v_test.txt', 'r') as f:
	data = f.read().splitlines()
Y_test = []
X_test = []
sentence_lengths_test = []
for line in data:
	label, name, doc_len, text = line.split('<fff>')
	Y_test.append(int(label))
	X_test.append(text.split()[:MAX_DOC_LENGTH])
	sentence_lengths_test.append(int(doc_len))
Y_test = np.array(Y_test)
X_test = np.array(X_test)
sentence_lengths_test = np.array(sentence_lengths_test)





class RNN_LSTM:
	def __init__(self, learning_rate, embedding_size, vob_size, lstm_size, max_doc_length, batch_size, pre_train_model):
		self.learning_rate = learning_rate
		self.embedding_size = embedding_size
		self.vob_size = vob_size
		self.lstm_size = lstm_size
		self.max_doc_length = max_doc_length
		self.batch_size = batch_size
		self.pre_train_model = pre_train_model

		self.X = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_doc_length])
		self.Y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
		self.sentence_lengths = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])

	def build_graph(self):
		embeddings = self.get_embedding_layer(self.X)
		h_vectors = self.LSTM_layer(embeddings)
		W = tf.get_variable(name='weights', shape=[self.lstm_size, NUM_CLASS], initializer=tf.random_normal_initializer())
		b = tf.get_variable(name='biases', shape=[NUM_CLASS], initializer=tf.random_normal_initializer())
		logits = tf.add(tf.matmul(h_vectors, W), b)
		sm = tf.nn.softmax(logits)
		y_pre = tf.argmax(sm, axis=1)
		labels = tf.one_hot(self.Y, NUM_CLASS)
		loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
		loss = tf.reduce_mean(loss)
		return loss, y_pre


	def get_embedding_layer(self, indices):
		#self.embeddings_matrix = tf.get_variable(name='embeddings_maxtrix', shape=[self.vob_size, self.embedding_size], initializer=tf.constant_initializer(self.pre_train_model))
		return tf.nn.embedding_lookup(self.pre_train_model, indices)

	def LSTM_layer(self, embeddings):
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
		lstm_input = tf.unstack(embeddings, axis=1) #input for timestep.
		zero_state = tf.zeros(shape=[self.batch_size, self.lstm_size])
		init_state = tf.nn.rnn_cell.LSTMStateTuple(zero_state, zero_state)
		output_time_step, _ = tf.nn.static_rnn(cell=lstm_cell, inputs=lstm_input, initial_state=init_state, sequence_length=self.sentence_lengths, dtype=tf.float32)
		output_batch = tf.stack(output_time_step, axis=1)
		output_averages = tf.reduce_sum(output_batch, axis=1) / tf.reshape(tf.cast(self.sentence_lengths, tf.float32), [-1, 1])
		return output_averages
	def trainner(self, loss):
		train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
		return train_op



class Dive_data:
	def __init__(self, X_train, Y_train, sentence_lengths, batch_size):
		self.epoch = 0
		self.X_train = X_train
		self.Y_train = Y_train
		self.sentence_lengths = sentence_lengths
		self.batch_size = batch_size
		self.batch_num = X_train.shape[0] / batch_size
		self._id = np.arange(X_train.shape[0])
		self._id_batch = np.split(self._id[:self._id.shape[0] - self._id.shape[0] % batch_size], self.batch_num)
		self._id_last_batch = self._id[self._id.shape[0] - self._id.shape[0] % batch_size : self._id.shape[0]]
		self.batch = 1


	def next_batch(self):
		if(self.batch == self.batch_num):
			self.epoch += 1
			#x_train, y_train, sentence_lengths_= self.X_train[self._id_last_batch], self.Y_train[self._id_last_batch], self.sentence_lengths[self._id_last_batch]
			self.batch = 1
			idx = np.random.permutation(self.X_train.shape[0])
			self.X_train = self.X_train[idx]
			self.Y_train = self.Y_train[idx]
			self.sentence_lengths = self.sentence_lengths[idx]
			return self.X_train[self._id_batch[self.batch - 2]], self.Y_train[self._id_batch[self.batch - 2]], self.sentence_lengths[self._id_batch[self.batch - 2]]
		self.batch += 1
		return self.X_train[self._id_batch[self.batch - 2]], self.Y_train[self._id_batch[self.batch - 2]], self.sentence_lengths[self._id_batch[self.batch - 2]]



def eval():
	epochs = 100
	batch_size = 128
	model = RNN_LSTM(0.001, embedding_size=EMBEDDING_SIZE, vob_size=VOB_SIZE, lstm_size=128, max_doc_length=MAX_DOC_LENGTH, batch_size=batch_size, pre_train_model=pre_train_model)
	loss, y_pre = model.build_graph()
	train_op = model.trainner(loss)
	train_data = Dive_data(X_train, Y_train, sentence_lengths, batch_size)
	test_data = Dive_data(X_test, Y_test, sentence_lengths_test, batch_size)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		i = 0
		while(i < 1000**2):
			i = i + 1
			x_train, y_train, sentence_lengths_ = train_data.next_batch()
			loss_train, train_o, y_train_pre_ = sess.run([loss, train_op, y_pre], feed_dict={
				model.X : x_train,
				model.Y : y_train,
				model.sentence_lengths: sentence_lengths_
				})
			
			if(i%20==0):
				print('loss: {}'.format(loss_train))
			if(train_data.batch == 1):
				k = 0
				number_true_pre = 0.
				while(k < test_data.batch_num):
					k = k + 1
					x_test, y_test, sentence_lengths_test_ = test_data.next_batch()
					y_pre_ = sess.run([y_pre], feed_dict={
						model.X : x_test,
						model.Y : y_test,
						model.sentence_lengths: sentence_lengths_test_
						})
					matches = np.equal(y_test, y_pre_)
					number_true_pre += np.sum(matches.astype(float))
				print('Epoch: {}, Acc_test: {}'.format(test_data.epoch, number_true_pre*1./ (test_data.batch_size * test_data.batch_num)))	

eval()


