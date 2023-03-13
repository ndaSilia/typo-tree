from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import re
from matplotlib import pyplot as plt


class DataBase():
    def __init__(self, seq_len=60, step=1):
        self.seq_len = seq_len
        self.step = step
        self.data = None
        self.words = None
        self.data_size = None
        self.vocab_size = None
        self.w2i = None
        self.i2w = None
        self.x = None
        self.y = None
        self.train_size = None
    
    def Normalize(self, path, pattern=None):
        data = open(path, 'r', encoding='utf-8').read()
        if pattern != None:
            data = re.sub(pattern, '', data)
        else:
            data = data.replace('\n', '\0')
            data = re.sub(r'[^\uAC00-\uD7A3 0-9a-zA-Z!?.,\'\"', '', data)
        words = sorted(list(set(data.lower())))
        self.data, self.words, self.data_size, self.vocab_size = data, words, len(data), len(words)
    
    def Dictionary(self, words=self.words, dic=None):
        if dic != None:
            w2i = open('dict.pickle', 'rb').pickle.load(f)
            i2w = dict()
            for w, i in w2i.items():
                i2w[i] = w
        else:
            w2i = dict((w, i) for i, w in enumerate(words))
            i2w = dict((i, w) for i, w in enumerate(words))
        self.w2i, self.i2w = w2i, i2w
    
    def Preprocessing(self, data=self.data, data_size=self.data_size, seq_len=self.seq_len, step=self.step):
        X = []
        Y = []
        for i in range(0, data_size - seq_len, step):
            X.append(data[i:i+seq_len])
            Y.append(data[i+seq_len])
        self.x, self.y, self.train_size = X, Y, len(X)
        self.seq_len, self.step = seq_len, step
    
    def Vectorize(self, train_size=self.train_size, seq_len=self.seq_len, vocab_size=self.vocab_size, data_size=self.data_size, x=self.x, y=self.y, w2i=self.w2i):
        X = np.zeros((train_size, seq_len, vocab_size))
        Y = np.zeros((train_size, vocab_size))
        for i, s in enumerate(x):
            for t, w in enumerate(s):
                X[i, t, w2i[w]] = 1
            Y[i, w2i[y[i]]] = 1
        self.x, self.y = X, Y
    
    def Encoder(self, code, seq_len):
        if len(code) > seq_len:
            code = code[len(code)-seq_len:len(code)]
        elif len(code) < seq_len:
            for _ in range(seq_len-len(code)):
                code = ' ' + code
        encode = np.zeros((1, seq_len, self.n_vocab))
        for t, word in enumerate(code):
            encode[0, t, self.word2int[word]] = 1.
        return ecode
    
    def Decoder(self, code, types, i2w=self.i2w):
        if types == 'random_by_confidence':
            chance = code[0]
            chance_sum = chance / np.sum(chance)
            decoded = np.random.choice(len(chance), p=chance_sum)
            return i2w[decoded]
        elif types == 'highest':
            decoded = np.argmax(code)
            return i2w[decoded]


class LSTMModel():
    def __init__(self, data=None, hidden_size=128, dropout=0.2):
        self.data = data
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.model = None
        self.callbacks = None
    
    def ModelBuild(self, hidden_size=self.hidden_size, seq_len=self.data.seq_len, vocab_size=self.data.vocab_size, dropout=self.dropout):
        model = Sequential()
        model.add(LSTM(hidden_size, input_shape=(seq_len, vocab_size), return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(hidden_size))
        model.add(Dropout(dropout))
        model.add(Dense(vocab_size, activation='softmax'))
        model.summary()
        filepath = "saved_weights/saved_weights-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='loss', metrics=['accuracy'], verbose=1, save_best_only=True, mode='min', save_freq=2500)
        callbacks = [checkpoint]
        self.model, self.callbacks = model, callbacks
    
    def ModelTrain(self, model=self.model, x=self.data.x, y=self.data.y, batch_size=self.hidden_size, epochs=50, callbacks=self.callbacks, save=True):
        self.history = model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        if save:
            model.save('typo_tree.h5')
    
    def Predict(self, data, model=self.model, seq_len=self.data.seq_len, types='random_by_confidence'):
        preds = model.predict(self.data.Encoder(data, seq_len), verbose=0)
        preds = self.data.Decoder(preds, types)
        return preds


path = '.data/txt'
data = DataBase()
data.Normalize(path)
data.Dictionary()
data.Preprocessing()
data.Vectorize()
model = LSTMModel(data)
model.ModelBuild()
model.ModelTrain()
user_input = input('Text: ')
seed = user_input
n_latter = 400
for i in range(n_latter):
    user_input += model.Predict(user_input)
print('Your seed: ' + seed + ' and model predict: ' + user_input[len(seed):])
