#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:18:16 2020

@author: leonidkotov
"""

import numpy as np
import matplotlib.pyplot as plt
from dataset import create_dataset
from subplots import plot_signal, plot_examples, plot_signals, plot_signals_with_linestyle
from autoencoders import simpleautoencoder, deepautoencoder
from scipy import signal


#%%

# path to your directory with dataset files
files_path = "data/" 
# 6 persons ( 1 - Андрей, 2 - АндрСеме, 3 - Лёня, 4 - Миша, 5 - Юра, 6 - Алексей Олегович )
train_persons = [1, 2, 4, 5]
test_persons = [3, 6]

persons = {
        1: "Андрей Лук",
        2: "Андрей Сем",
        3: "Лёня",
        4: "Миша",
        5: "Юра",
        6: "Алексей",
        }

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

signal_len = 400

#%%

# Load dataset from directory files_path
# Создание датасета (обучение и тестовые) из файлов в директории files_path
# Для автокодировщика не требуется validation dataset
(train_signals, train_labels, train_positions) = create_dataset(files_path, train_persons, randomize=False)
(test_signals, test_labels, test_positions) = create_dataset(files_path, test_persons, randomize=True)

# Print shape of 
print("Train signal shape: " + str(train_signals.shape))
print("Train labels shape: " + str(train_labels.shape))
print("Test signal shape: " + str(test_signals.shape))
print("Test labels shape: " + str(test_labels.shape))

print("Load data successful")

#%%
# Normalization of dataset
# Нормализация датасета

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!! Под вопросом, потому что максимальное значение взято на глаз по графику !!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

train_signals = train_signals / 120
test_signals = test_signals / 120

#%%

# Print one signal from each person[class]
fig, axs = plt.subplots(len(train_persons), len(classes), figsize=(25, 9))

for p in range(len(train_persons)):
    for c in range(len(classes)):
        pos = train_positions[train_persons[p]][c]
        axs[p, c].plot(train_signals[pos])
        axs[p, c].set(xlabel = classes[c], ylabel = persons[train_persons[p]])
        axs[p, c].tick_params('x', labelrotation=45)
#        for label in axs[p, c].get_xticklabels():
#            print(label)
        
for ax in axs.flat:
    ax.label_outer()

        
plt.show()   

#%%

#autoencoder, encoder, decoder = simpleautoencoder(signal_len)
autoencoder, encoder, decoder = deepautoencoder(signal_len)

print(autoencoder.summary())

autoencoder.compile(optimizer = 'adam', loss = 'mse', metrics = ["accuracy"])

autoencoder.fit(train_signals, train_signals,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(test_signals, test_signals))

#%%
encoded_signals = encoder.predict(test_signals, batch_size=256)
decoded_signals = decoder.predict(encoded_signals, batch_size=256)

plot_examples(test_signals, encoded_signals, colors = ['r', 'g'])

plot_examples(test_signals, decoded_signals)

score = autoencoder.evaluate(x = test_signals, y = test_signals, verbose = 0)
print(score)
print('Test accuracy:', score[1])

y_test = []
for i in test_labels:
    y_test.append(np.argmax(i))

#%%
    
# Need learning it from official KERAS AUTOENCODER documentation
plt.figure(figsize=(6, 6))
plt.scatter(encoded_signals[:, 0], encoded_signals[:, 1], c=y_test)
plt.colorbar()
plt.show()


#%%

# Number of signal for review
sig = 1

diffXbetY = []
for i in range(len(test_signals[sig])):
    diffXbetY.append(abs(test_signals[sig, i] - decoded_signals[sig, i]))

plot_signals_with_linestyle([test_signals[sig], decoded_signals[sig], diffXbetY], ['r','b', 'orange'], ['вход', 'выход', 'разница'], ["-","-","--"])

#%%
# Welch for difference between test and decoded signals
frex, Pxx = signal.welch(test_signals[sig])
frey, Pyy = signal.welch(decoded_signals[sig])
frew, Pyw = signal.welch(diffXbetY)

fig2 = plt.figure(figsize=(12, 6))
plt.semilogy(frex, Pxx, label = "Исходный тестовый сигнал")
plt.semilogy(frey, Pyy, label = "Декодированный тестовый сигнал")
plt.semilogy(frew, Pyw, label = "Разница между тестовым и декодированным сигналом")
plt.grid(True)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD')
plt.legend()
plt.show()

