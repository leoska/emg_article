# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:56:58 2019

@author: leoska
"""

import pickle
import numpy as np
from random import shuffle
from tensorflow.python.keras.utils import to_categorical

def create_dataset(file_path, persons, validation_split = 0):
    path = file_path + "{}_{}.txt"
    sgn = []
    lbl = []
    for i in persons:
        for j in range(9):
            print("person: %d; class: %d; position: %d" % (i, j, len(lbl)))
            with open(path.format(i, j + 1), "rb") as fp:  # Unpickling
                data = pickle.load(fp)

            for k in range(np.shape(data)[0]):
                sgn.append(data[k])
                lbl.append(j)

    sgn = np.asarray(sgn, dtype=np.float32)
    lbl = np.asarray(lbl, dtype=np.int32)

    c = list(zip(sgn, lbl))
    #shuffle(c)
    sgn, lbl = zip(*c)

    sgn = np.asarray(sgn, dtype=np.float64)
    lbl = np.asarray(lbl, dtype=np.int64)
    
    train_split = 1 - validation_split

    train_signals = sgn[0:int(train_split * len(sgn))]
    train_labels = lbl[0:int(train_split * len(lbl))]
    val_signals = sgn[int(train_split*len(sgn)):]
    val_labels = lbl[int(train_split*len(lbl)):]
    
#    nrows, ncols = train_signals.shape
#    train_signals = train_signals.reshape(nrows, ncols, 1)
#    nrows, ncols = val_signals.shape
#    val_signals = val_signals.reshape(nrows, ncols, 1)
#    nrows, ncols = test_signals.shape
#    test_signals = test_signals.reshape(nrows, ncols, 1)
    
    train_labels = to_categorical(train_labels)
    if (validation_split > 0):
        val_labels = to_categorical(val_labels)
#    test_labels = to_categorical(test_labels)

    return train_signals, train_labels, val_signals, val_labels