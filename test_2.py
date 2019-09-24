import torch
import torch.nn as nn
from torch.autograd import Variable


USE_CUDA = torch.cuda.is_available()

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, lstm_unit, max_doc_length, num_classes, batch_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_unit = lstm_unit
        self.max_doc_length = max_doc_length
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.lstm_unit, num_layers=1)
        self.linear = nn.Linear(self.lstm_unit, self.num_classes)

    def init_hidden(self):
        hidden = torch.Variable(torch.zeros(1, self.batch_size, self.lstm_unit)).cuda() if USE_CUDA else torch.Variable(torch.zeros(1, self.batch_size, self.lstm_unit))
        context = torch.Variable(torch.zeros(1, self.batch_size, self.lstm_unit)).cuda() if USE_CUDA else torch.Variable(torch.zeros(1, self.batch_size, self.lstm_unit))
        return (hidden, context)

    def forward(self, x):
        x = self.embedding(x)
        self.hidden = self.init_hidden()
        output, self.hidden = self.lstm(x, self.hidden)
        
