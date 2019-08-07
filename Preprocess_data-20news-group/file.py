import numpy as np 
import os
from nltk.stem.porter import PorterStemmer
import re
from collections import defaultdict


with open('stopwords.txt', 'r', errors='ignore') as f:
	stopwords = f.read().split()

root = '../dataset/'
data_types = os.listdir(root)
train_dir, test_dir = (data_types[1], data_types[0]) if 'train' in data_types[1] else (data_types[0], data_types[1])

news_group_list = [group_name for group_name in os.listdir(root + test_dir)]
news_group_list.sort()
stemmer = PorterStemmer()

'''

doc_list = []
for label, file_names in enumerate(news_group_list):
	files = os.listdir(root + train_dir + '/' + file_names)
	files.sort()
	for file in files:
		with open(root + train_dir + '/' + file_names + '/'+ file, 'r', errors='ignore') as f:
			doc = f.read().lower()
			words = [stemmer.stem(word) for word in re.split('\W+', doc) if word not in stopwords]
			content = ' '.join(words)
			processed_doc = str(label) + '<fff>' + file + '<fff>' + content
			doc_list.append(processed_doc)

with open(root + train_dir + '/' +'doc_train_lists.txt', 'w', errors='ignore') as f:
	for item in doc_list:
		f.write('{}\n'.format(item))



def compute_idf(df, corpus_size):
	return np.log10(corpus_size * 1. / df)

def generate_vocabulary():
	vob_dict = defaultdict(int)
	with open(root + train_dir + '/' + 'doc_train_lists.txt', 'r', errors='ignore') as f:
		corpus = f.read().splitlines()
	corpus_size = len(corpus)
	for doc in corpus:
		words = doc.split('<fff>')[2].split()
		words_set = list(set(words))
		for word in words_set:
			vob_dict[word] += 1
	print(len(vob_dict.keys()))
	words_idf = [(word, compute_idf(vob_dict[word], corpus_size)) for word in vob_dict.keys() if vob_dict[word] > 10]
	#words_idf.sort(key=lambda (word, idf):-idf)
	with open(root + train_dir + '/' + 'vobcabulary.txt', 'w', errors='ignore') as f:
		for word, idf in words_idf:
			f.write('{}<fff>{}\n'.format(word, idf))
generate_vocabulary()
'''
'''

'''
'''
def get_tf_idfs():
	with open(root + test_dir + '/' + 'doc_test_lists.txt', 'r') as f:
		corpus = f.read().splitlines()
	with open(root + train_dir + '/' + 'vobcabulary.txt', 'r') as f:
		word_idfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1])) for line in f.read().splitlines()]

	word_ids = dict([(word, index) for index, (word, idf) in enumerate(word_idfs)])
	idfs = dict(word_idfs)
	corpus_tf_idfs = []
	for doc in corpus:
		features = doc.split('<fff>')
		words = [word for word in features[2].split() if word in idfs]
		words_set = list(set(words))
		max_tf = max([words.count(word) for word in words_set])
		sum_square = 0.0
		words_tfidfs = []

		for word in words_set:
			tf_idf = words.count(word) * 1. / max_tf * idfs[word]
			sum_square += tf_idf**2
			words_tfidfs.append((word_ids[word], tf_idf))
		tf_idf_normalize = [str(index) + ':' + str(float(tf_idf_value) / sum_square) for index, tf_idf_value in words_tfidfs]
		content = ' '.join(tf_idf_normalize)
		prs = str(features[0]) + '<fff>' + features[1] + '<fff>' + content
		corpus_tf_idfs.append(prs)

	with open(root + test_dir+ '/' + 'encode.txt', 'w') as f:
		for item in corpus_tf_idfs:
			f.write('{}\n'.format(item))

get_tf_idfs()
'''

'''

for doc in corpus:
	dict_ = defaultdict(int)
	words = doc.split()
	for word in words:
		if(word not in dict_.keys()):
			vob[word] +=1
		dict_[word] += 1
	lists_dict.append(dict_)
print(vob)

'''
'''
with open('glove.6B.300d.txt', 'r') as f:
	lines = f.read().splitlines()

words_glove = []
for line in lines:
	word = line.split()[0]
	words_glove.append(word)
'''
'''
doc_list = []
for label, file_names in enumerate(news_group_list):
	files = os.listdir(root + test_dir + '/' + file_names)
	files.sort()
	for file in files:
		with open(root + test_dir + '/' + file_names + '/'+ file,'r', errors='ignore') as f:
			doc = f.read().lower()
			words = [word for word in re.split('\W+', doc) ]
			content = ' '.join(words)
			processed_doc = str(label) + '<fff>' + file + '<fff>' + content
			doc_list.append(processed_doc)

with open(root + test_dir + '/' +'doc_lists.txt', 'w') as f:
	for item in doc_list:
		f.write('{}\n'.format(item))

'''
'''
with open(root + train_dir + '/' + 'doc_lists.txt') as f:
	lines = f.read().splitlines()
vob_dict = defaultdict(int)
for line in lines:
	text = line.split('<fff>')[2]
	words = text.split()
	for word in words:
		vob_dict[word] += 1

vob = [word for word, feq in zip(vob_dict.keys(), vob_dict.values()) if vob_dict[word] > 10 and word in words_glove and word != 'unknown' and word != 'pad']
vob.sort()
vob = [(word, index + 2) for index, word in enumerate(vob)]
with open(root + train_dir + '/' + 'vob_w2v.txt', 'w') as f:
	f.write('unknown:0\n')
	f.write('pad:1\n')
	for word, index in vob:
		f.write('{}:{}\n'.format(word,index))
'''


with open(root + train_dir + '/' + 'vob_w2v.txt', 'r', errors='ignore') as f:
	lines = f.read().splitlines()
vob = dict([(line.split(':')[0], line.split(':')[1]) for line in lines])

with open(root + test_dir +'/'+'doc_lists.txt', 'r', errors='ignore') as f:
	documents = f.read().splitlines()
doc_encode = []
for document in documents:
	label, name, text = document.split('<fff>')
	words = text.split()
	words_id = []
	for word in words:
		if word in vob.keys():
			words_id.append(vob[word])
		else:
			words_id.append(vob['unknown'])
	doc_length = 0
	if len(words_id) < 500:
		doc_length = len(words_id)
		for _ in range(500 - len(words_id)):
			words_id.append(vob['pad'])
	else:
		doc_length = 500
	content = ' '.join(words_id)
	doc_encode.append(str(label) + '<fff>' + name + '<fff>' + str(doc_length) + '<fff>' + content)

with open(root + test_dir + '/' + 'encode_w2v_test.txt', 'w') as f:
	for item in doc_encode:
		f.write('{}\n'.format(item))






	
