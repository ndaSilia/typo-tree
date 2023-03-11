from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
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
        y = np.zeros((self.train_size, self.n_vocab))
        for i, s in enumerate(self.train_x):
            for t, word in enumerate(s):
                x[i, t, self.word2int[word]] = 1
            y[i, self.word2int[self.train_y[i]]] = 1
        return x, y
    
    def Decoder(self, code):
        code = np.asarray(code)
        code = np.log(code)
        exp_code = np.exp(code)
        code = exp_code / np.sum(exp_code)
        chance = np.random.multinomial(1, code, 1)
        return np.argmax(chance)


class LSTMModel():
    def __init__(self, hidden_size, dropout_rate, path, seq_len, step=1):
        self.data = DataSet(path, seq_len, step)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.model, self.callbacks = self.ModelBuild()
        self.x = self.data.x
        self.y = self.data.y
        self.history = list()
    
    def ModelBuild(self):
        model = Sequential()
        model.add(LSTM(self.hidden_size, input_shape=(self.data.seq_length, self.data.n_vocab), return_sequences=True))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(self.hidden_size))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.data.n_vocab, activation='softmax'))
        optimizer = RMSprop(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        filepath = "saved_weights/saved_weights-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='loss', metrics=['accuracy'], verbose=1, save_best_only=True, mode='min', save_freq=2500)
        callbacks_list = [checkpoint]
        return model, callbacks_list
    
    def Train(self, batch_size, epochs):
        self.history = self.model.fit(self.x, self.y, batch_size=batch_size, epochs=epochs, callbacks=self.callbacks)
    
    def Prediction(self, code):
        if len(code) > self.seq_length:
            code = code[len(code)-self.seq_length:len(code)]
        elif len(code) < self.seq_length:
            for _ in range(self.seq_length-len(code)):
                code = ' ' + code
        x_pred = np.zeros((1, self.seq_length, self.n_vocab))
        for t, word in enumerate(code):
            x_pred[0, t, self.data.word2int[char]] = 1.
        preds = self.model.predict(x_pred, verbose=0)
        next_word = self.data.int2word[self.data.Decoder(preds)]
        return next_word
    
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

__path__ = './data.txt'
models = LSTMModel(128, 0.2, __path__, 200)
models.Train(128, 100)
user_input = input('')
gen = user_input
n_letter = 400
for i in range(n_letter):
    pred = models.Prediction(user_input)
    gen += pred
    user_input = user_input[1:] + pred
