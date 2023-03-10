from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.optimizers import RMSprop
import numpy as np
import random
import sys
import re
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt


class DataSet():
    def __init__(self, path, seq_length, step):
        self.path = path
        self.data, self.words = self.Normalize()
        self.word2int = self.Dictionaries(0)
        self.int2word = self.Dictionaries(1)
        self.n_words, self.n_vocab = self.Summary()
        self.seq_length = seq_length
        self.step = step
        self.train_x, self.train_y, self.train_size = self.SetInput()
        self.x, self.y = self.Encoder()
    
    def Normalize(self):
        data = open(self.path, 'r', encoding='utf-8').read()
        data = re.sub(r'[^\uAC00-\uD7A3 0-9a-zA-Z?!.,]', '', data)
        words = sorted(list(set(data.lower())))
        return data, words
    
    def Dictionaries(self, types):
        if types == 0:
            return dict((w, i) for i, w in enumerate(self.words))
        elif types == 1:
            return dict((i, w) for i, w in enumerate(self.words))
    
    def Summary(self):
        return len(self.data), len(self.words)
    
    def SetInput(self):
        sentences = []
        next_words = []
        for i in range(0, self.n_words - self.seq_length, self.step):
            sentences.append(self.data[i:i+self.seq_length])
            next_words.append(self.data[i+self.seq_length])
        train_size = len(sentences)
        return sentences, next_words, train_size
    
    def Encoder(self):
        x = np.zeros((self.train_size, self.seq_length, self.n_vocab))
        y = np.zeros((self.train_size, n_vocab))
        for i, s in enumerate(self.train_x):
            for t, word in enumerate(s):
                x[i, t, self.word2int[word]] = 1
            y[i, self.word2int[self.train_y[i]]] = 1
        return x, y


class LSTMModel():
    def __init__(self, hidden_size, dropout_rate, epochs, batch_size, path, seq_len, step=0):
        self.data = DataSet(path, seq_len, step)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.model, self.callbacks = self.ModelBuild()
        self.epochs = epochs
        self.batch_size = batch_size
        self.x = self.data.x
        self.y = self.data.y
        self.history = list()
    
    def ModelBuild(self):
        model = Sequential()
        model.add(LSTM(self.hidden_size, input_shape=(seq_length, n_vocab), return_sequences=True))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(self.hidden_size))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(n_vocab, activation='softmax'))
        optimizer = RMSprop(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        filepath = "saved_weights/saved_weights-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='loss', metrics=['accuracy'], verbose=1, save_best_only=True, mode='min', save_freq=2500)
        callbacks_list = [checkpoint]
        return model, callbacks_list
    
    def Train(self):
        self.history = self.model.fit(self.x, self.y, batch_size=self.batch_size, epochs=self.epochs, callbacks=self.callbacks)
    
    def ShowPlt(self, types):
        if types == 'loss':
            loss = history.history['loss']
            epochs = range(1, len(loss) + 1)
            plt.plot(epochs, loss, 'y', label='Training loss')
            plt.title('Training loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
