from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import re
from matplotlib import pyplot as plt


class DataBase():
    def __init__(self, path, seq_len, step):
        self.data, self.words = self.Normalize(path)
        self.word2int, self.int2word = self.Dictionary()
        self.n_words, self.n_vocab = len(self.data), len(self.words)
        self.x, self.y, self.train_size = self.Vectorize(seq_len, step)
        self.seq_len = seq_len
    
    def Normalize(self, path):
        data = open(path, 'r', encoding='utf-8').read()
        data = re.sub(r'[^\uAC00-\uD7A3 0-9a-zA-Z?!.,]', '', data)
        words = sorted(list(set(data.lower())))
        return data, words
    
    def Dictionary(self):
        w2i = dict((w, i) for i, w in enumerate(self.words))
        i2w = dict((i, w) for i, w in enumerate(self.words))
        return w2i, i2w
    
    def Vectorize(self, seq_len, step):
        x = []
        y = []
        for i in range(0, self.n_words - seq_len, step):
            x.append(self.data[i:i+seq_len])
            y.append(self.data[i+seq_len])
        train_size = len(x)
        X = np.zeros((train_size, seq_len, self.n_vocab))
        Y = np.zeros((train_size, self.n_vocab))
        for i, s in enumerate(x):
            for t, w in enumerate(s):
                X[i, t, self.word2int[w]] = 1
            Y[i, self.word2int[y[i]]] = 1
        return X, Y, train_size
    
    def Encoder(self, code, seq_len):
        if len(code) > seq_len:
            code = code[len(code)-seq_len:len(code)]
        elif len(code) < seq_len:
            for _ in range(seq_len-len(code)):
                code = ' ' + code
        encode = np.zeros((1, seq_len, self.n_vocab))
        for t, word in enumerate(code):
            encode[0, t, self.word2int[word]] = 1.
        return encode
    
    def Decoder(self, code, types):
        if types == 'random_by_confidence':
            chance = code[0]
            chance_sum = chance / np.sum(chance)
            decoded = np.random.choice(len(chance), p=chance_sum)
            return self.int2word[decoded]
        elif types == 'highest':
            decoded = np.argmax(code)
            return self.int2word[decoded]
            


class LSTMModel():
    def __init__(self, path, seq_len=60, step=1, hidden_size=128, dropout=0.2):
        self.data = DataBase(path, seq_len, step)
        self.x, self.y = self.data.x, self.data.y
        self.model, self.callbacks = self.ModelBuild(hidden_size, seq_len, self.data.n_vocab, dropout)
        self.history = list()
    
    def ModelBuild(self, hidden_size, seq_len, n_vocab, dropout):
        model = Sequential()
        model.add(LSTM(hidden_size, input_shape=(seq_len, n_vocab), return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(hidden_size))
        model.add(Dropout(dropout))
        model.add(Dense(n_vocab, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
        model.summary()
        filepath = "saved_weights/saved_weights-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='loss', metrics=['accuracy'], verbose=1, save_best_only=True, mode='min', save_freq=2500)
        callbacks = [checkpoint]
        return model, callbacks
    
    def ModelTrain(self, batch_size, epochs, save=True):
        self.history = self.model.fit(self.x, self.y, batch_size=batch_size, epochs=epochs, callbacks=self.callbacks)
        if save:
            self.model.save('typo_tree.h5')
    
    def Predict(self, input_data, types='random_by_confidence'):
        preds = self.model.predict(self.data.Encoder(input_data, self.data.seq_len), verbose=0)
        preds = self.data.Decoder(preds, types)
        return preds
    
    def ShowPlot(types='loss'):
        if types == 'loss':
            loss = history.history['loss']
            epochs = range(1, len(loss) + 1)
            plt.plot(epochs, loss, 'y', label='Training loss')
            plt.title('Training loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()


__path__ = './data.txt'
model = LSTMModel(__path__, 140)
model.ModelTrain(128, 20)
user_input = input('Text: ')
seed = user_input
n_latter = 400
for i in range(n_latter):
    user_input += model.Predict(user_input)
print('Your seed: ' + seed + ' and model predict: ' + user_input[len(seed):])
