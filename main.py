#!/usr/bin/env python3
# encoding: utf-8
# Author: Linus Boyle <linusboyle@gmail.com> 

import keras
from keras.layers import Embedding, Dense, Dropout, Flatten, Input, MaxPooling1D, Conv1D, concatenate
from keras.layers.recurrent import LSTM
from keras.models import Model, load_model
from keras.utils import to_categorical, plot_model
from keras.optimizers import SGD, Adam
from keras.initializers import random_normal
import keras.backend as T
import os

# input config
vocab_size = 59550
vec_dim = 300
word_len = 1000

# output config
output_len = 8

#fit
batch_size=256
epochs=5

def my_acc(y_true, y_pred):
    return T.mean(T.equal(T.argmax(y_true), T.argmax(y_pred)))

import pickle
import gzip

def load_data(x_file = 'x.pkl.gz', y_file='y.pkl.gz'):
    f_x = gzip.open(x_file, 'rb')
    x_train, x_test = pickle.load(f_x)
    f_x.close()

    f_y = gzip.open(y_file, 'rb')
    y_train, y_test = pickle.load(f_y)
    f_y.close()

    return ((x_train, to_categorical(y_train)), (x_test, to_categorical(y_test)))

def construct_cnn(maxlen=word_len, max_features=vocab_size, embed_size=vec_dim):
    input_seq = Input(shape=[maxlen], name='x_seq')

    embed_matrix = Embedding(max_features, embed_size, embeddings_initializer=random_normal(mean=0, stddev=1/(max_features*embed_size)))(input_seq)

    convs = []
    filter_sizes = [2, 3, 4, 5]
    for kernel_h in filter_sizes:
        l_conv = Conv1D(filters=50, kernel_size=kernel_h, activation='relu')(embed_matrix)
        l_pool = MaxPooling1D(maxlen - kernel_h + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    dropped = Dropout(0.35)(merge)
    densed = Dense(32, activation='relu')(dropped)
    output = Dense(units=8, activation='softmax')(densed)

    model = Model([input_seq], output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.002), metrics=["acc"])
    return model

def construct_mlp(maxlen=word_len, max_features=vocab_size, embed_size=vec_dim):
    input_seq = Input(shape=[maxlen], name='x_seq')

    embed_matrix = Embedding(max_features, embed_size, embeddings_initializer=random_normal(mean=0, stddev=1/(max_features*embed_size)))(input_seq)
    flattened = Flatten()(embed_matrix)

    densed = Dense(128, activation='relu')(flattened)
    dropped = Dropout(0.35)(densed)
    output = Dense(units=8, activation='softmax')(densed)

    model = Model([input_seq], output)
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.002), metrics=["acc"])
    return model

def construct_rnn(maxlen=word_len, max_features=vocab_size, embed_size=vec_dim):
    input_seq = Input(shape=[maxlen], name='x_seq')

    embed_matrix = Embedding(max_features, embed_size, embeddings_initializer=random_normal(mean=0, stddev=1/(max_features*embed_size)), mask_zero=True)(input_seq)
    lstm = LSTM(64)(embed_matrix)

    dropped = Dropout(0.5)(lstm)
    densed = Dense(32, activation='relu')(dropped)
    output = Dense(units=8, activation='softmax')(densed)

    model = Model([input_seq], output)
    model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=["acc"])
    return model

if __name__ == "__main__":
    train_d, test_d = load_data()
    x_train, y_train = train_d
    x_test, y_test = test_d

    if os.path.exists('./model.h5'):
        model = load_model('./model.h5')
    else:
        model = construct_cnn()

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            #validation_data=(x_test, y_test))
            validation_split=0.1)
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
    model.save('model.h5')
    plot_model(model, to_file='arch.png', show_shapes=True)
