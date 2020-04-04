# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:56:58 2019

@author: leoska
"""

import pickle
import numpy as np
from random import shuffle
from tensorflow.keras.utils import to_categorical

def load_from_file(file_path, persons, randomize):
    path = file_path + "{}_{}.txt"
    sgn = []
    lbl = []
    pos = {}
    for i in persons:
        pos[i] = []
        for j in range(9):
            pos[i].append(len(lbl))
#            print("person: %d; class: %d; position: %d" % (i, j, len(lbl)))
            with open(path.format(i, j + 1), "rb") as fp:  # Unpickling
                data = pickle.load(fp)

            for k in range(np.shape(data)[0]):
                sgn.append(data[k])
                lbl.append(j)
                
    sgn = np.asarray(sgn, dtype=np.float32)
    lbl = np.asarray(lbl, dtype=np.int32)

    c = list(zip(sgn, lbl))
    if randomize:
        shuffle(c)
    sgn, lbl = zip(*c)
    
    sgn = np.asarray(sgn, dtype=np.float64)
    lbl = np.asarray(lbl, dtype=np.int64)
    
    return (sgn, lbl, pos)

def create_dataset(file_path: str, persons, randomize: bool = True):
    '''Create dataset from text files in file_path directory

    Args:
        file_path (string): The amount of distance traveled
        persons (array): 
        randomize (bool): Should the fuels refilled to cover the distance?

    Raises:
        RuntimeError: Out of fuel

    Returns:
        cars: A car mileage
    '''
    (sgn, lbl, positions) = load_from_file(file_path, persons, randomize)

    train_signals = sgn[0:len(sgn)]
    train_labels = lbl[0:len(lbl)]
    
    train_labels = to_categorical(train_labels)

    return (train_signals, train_labels, positions)



def create_dataset_with_validation(file_path, persons, randomize = False, validation_split = 0):
    (sgn, lbl) = load_from_file(file_path, persons, randomize)
    
    train_split = 1 - validation_split

    train_signals = sgn[0:int(train_split * len(sgn))]
    train_labels = lbl[0:int(train_split * len(lbl))]
    val_signals = sgn[int(train_split*len(sgn)):]
    val_labels = lbl[int(train_split*len(lbl)):]

    
    train_labels = to_categorical(train_labels)
    if (validation_split > 0):
        val_labels = to_categorical(val_labels)

    return (train_signals, train_labels), (val_signals, val_labels)


def load_from_file_with_index(file_path, persons, randomize):
    path = file_path + "{}_{}.txt"
    subjects = []
    for i in persons:
        subjects.append([])
        for j in range(9):
            subjects[i - 1].append([])
            with open(path.format(i, j + 1), "rb") as fp:  # Unpickling
                data = pickle.load(fp)

            for k in range(np.shape(data)[0]):
                subjects[i - 1][j].append(data[k])
    
    return subjects