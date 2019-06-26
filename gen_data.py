#!/usr/bin/env python3
# encoding: utf-8
# Author: Linus Boyle <linusboyle@gmail.com> 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re

wordlist_len = 60744

def get_text():
    def get_word_list(filename):
        print('processing ' + filename)

        texts = []
        with open(filename, 'r') as f_in:
            lines = f_in.read().strip().split('\n')
            for line in lines:
                emotions, words = line.split('\t')[1:]
                texts.append(words)
            f_in.close()
        return texts
    return (get_word_list('sinanews.train'), get_word_list('sinanews.test'))

def gen_x():
    tok = Tokenizer()
    train_x, test_x = get_text()
    tok.fit_on_texts(train_x)
    tok.fit_on_texts(test_x)
    x_train_seq = tok.texts_to_sequences(train_x)
    x_test_seq = tok.texts_to_sequences(test_x)
    x_train = pad_sequences(x_train_seq, maxlen=1000, value=0)
    x_test = pad_sequences(x_test_seq, maxlen=1000, value=0)
    return (x_train, x_test)

def gen_y():
    def get_target_y(filename):
        print('processing ' + filename)

        y = []
        with open(filename, 'r') as f_in:
            lines = f_in.read().strip().split('\n')
            for line in lines:
                emotion_buffer = line.split('\t')[1]
                emotions = emotion_buffer.split(' ')[1:]
                emotion_processed = np.argmax([int(re.match('.*:(\d*)', eToken).groups()[0]) for eToken in emotions])
                y.append(emotion_processed)
            f_in.close()
        return np.asarray(y)
    return (get_target_y('sinanews.train'), get_target_y('sinanews.test'))

import pickle
import gzip

if __name__ == '__main__':
    pickle.dump(gen_x(), gzip.open('x.pkl.gz', 'wb'))
    pickle.dump(gen_y(), gzip.open('y.pkl.gz', 'wb'))
