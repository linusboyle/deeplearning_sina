#!/usr/bin/env python3
# encoding: utf-8
# Author: Linus Boyle <linusboyle@gmail.com> 

import keras
from main_keras import load_data
from keras.models import load_model
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
import numpy as np

if __name__ == "__main__":
    train_data, test_data = load_data()
    x_test, y_test = test_data

    model = load_model('./model.h5')
    # acc
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))

    y_pred = model.predict(x_test)

    # f-score
    y_label_true = np.argmax(y_test, axis=1)
    y_label_pred = np.argmax(y_pred, axis=1)
    print("f1 score:(macro)"
    print(f1_score(y_label_true, y_label_pred, average='macro'))

    # coef
    coefs = []
    t_len = len(y_test)
    for i in range(t_len):
        coefs.append(pearsonr(y_test[i], y_pred[i]))
    print("coef:")
    print(np.mean(coefs))
