#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:18:16 2020

@author: leonidkotov
"""

import numpy as np
from dataset import create_dataset
from subplots import plot_signal
from tensorflow.keras import utils

#%%

# path to your directory with dataset files
files_path = "data/" 
# 6 persons ( 1 - Андрей, 2 - АндрСеме, 3 - Лёня, 4 - Миша, 5 - Юра, 6 - Алексей Олегович )
persons = [1, 2, 4, 5, 6]

# У каждого субъекта по девять жестов. 
#1) сгибание указательного пальца; 
#2) среднего; 
#3) безымянного; 
#4) поворот кисти влево; 
#5) вправо; 
#6) кисть вверх; 
#7) вниз; 
#8) щелчок большим с средним; 
#9) сжатие в кулак
classes = [
        "сгибание указательного пальца", 
        "сгибание среднего пальца", 
        "сгибание безымянного пальца", 
        "поворот кисти влево", 
        "поворот кисти вправо", 
        "кисть вверх",
        "кисть вниз",
        "щелчок большим с средним",
        "сжатие в кулак"
        ]

#%%

# Load dataset from files_path
(train_signals, train_labels) = create_dataset(files_path, persons)

# Print shape of 
print("Train signal shape: " + str(train_signals.shape))
print("Train labels shape: " + str(train_labels.shape))
#print("Test signal shape: " + str(test_signals.shape))
#print("Test labels shape: " + str(test_labels.shape))
#print("Val signal shape: " + str(val_signals.shape))
#print("Val labels shape: " + str(val_labels.shape))

print("Load data successful")

#%%
example_class = np.argmax(train_labels[165])
plot_signal(train_signals[165], label = classes[example_class], title = "Пример тренировочного сигнала")
