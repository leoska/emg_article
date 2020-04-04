#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:18:16 2020

@author: leonidkotov
"""

#Нампай
#Сайпай
#bottleneck layer - пережатый слой

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from dataset import create_dataset, load_from_file_with_index
from subplots import plot_signal, plot_examples, plot_signals, plot_signals_with_linestyle
from autoencoders import simpleautoencoder, deepautoencoder, deepautoencoder_work
import scipy.io
from scipy import signal
from sklearn import preprocessing
from fir_filter import get_filtered_signals


#%%
def rms(x):
    return np.sqrt(np.mean(x**2))


#%%
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


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

train_signals_pre = preprocessing.normalize(train_signals)
test_signals_pre = preprocessing.normalize(test_signals)

train_signals = get_filtered_signals(train_signals_pre)
test_signals = get_filtered_signals(test_signals_pre)

#%%

# Print one signal from each person[class] from train_signals
fig, axs = plt.subplots(len(train_persons), len(classes), figsize=(25, 9))
#plt.title("Пример каждого класса движения каждого субъекта тренировочных данных")

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

# Print one signal from each person[class] from test_signals
fig, axs = plt.subplots(len(test_persons), len(classes), figsize=(25, 5))
#plt.title("Пример каждого класса движения каждого субъекта тестовых данных")

for p in range(len(test_persons)):
    for c in range(len(classes)):
        pos = test_positions[test_persons[p]][c]
        axs[p, c].plot(test_signals[pos])
        axs[p, c].set(xlabel = classes[c], ylabel = persons[test_persons[p]])
        axs[p, c].tick_params('x', labelrotation=45)
#        for label in axs[p, c].get_xticklabels():
#            print(label)
        
for ax in axs.flat:
    ax.label_outer()


plt.show() 

#%%

#autoencoder, encoder, decoder = simpleautoencoder(signal_len)
autoencoder, encoder, decoder = deepautoencoder_work(signal_len)

print(encoder.summary())
print(decoder.summary())
print(autoencoder.summary())


autoencoder.compile(optimizer = 'adam', loss = 'mse', metrics = ["accuracy"])

history = autoencoder.fit(train_signals, train_signals,
                epochs=25,
                batch_size=256,
                shuffle=True,
                #validation_data=(test_signals, test_signals),
                validation_split=0.2,
                )

#%%
encoded_signals = encoder.predict(test_signals, batch_size=256)

# encoded_signals[:, 8:8] = 0

decoded_signals = decoder.predict(encoded_signals, batch_size=256)

plot_examples(test_signals, encoded_signals, colors = ['r', 'g'])

plot_examples(test_signals, decoded_signals)

score = autoencoder.evaluate(x = test_signals, y = test_signals, verbose = 0)
print(score)
print('Test accuracy (доля верных ответов): ', round(score[1] * 100, 4))

y_test = []
for i in test_labels:
    y_test.append(np.argmax(i))
    
    
# Need learning it from official KERAS AUTOENCODER documentation
#plt.figure(figsize=(6, 6))
#plt.scatter(encoded_signals[:, 0], encoded_signals[:, 1], c=y_test)
#plt.colorbar()
#plt.show()
    
#%%
    
# Графики суммы коэффициентов нейронов в bottleneck слое
x_inputs = np.eye(8, dtype=np.float64)
x_result = decoder.predict(x_inputs, batch_size=256)

for i in range(8):
    plt.figure(figsize=(15, 7))
    plt.plot(x_result[i])
    plt.title("Сумма коэффициентов " + str(i) + " нейрона")
    plt.show()

#%%

# 1 график суммы коэффициентов
fig, axs = plt.subplots(2, 8, figsize=(40, 6))

for p in range(8):
    axs[0, p].plot(x_inputs[p], color="orange")
    axs[1, p].plot(x_result[p])
    axs[1, p].set(xlabel = "Сумма коэффициентов " + str(p) + " нейрона")
    axs[1, p].tick_params('x', labelrotation=45)

for ax in axs.flat:
    ax.label_outer()


plt.show()   

#%%

# Root Mean Square
diffs = []

for i in range(len(test_signals)):
    diffs.append(test_signals[i] - decoded_signals[i]) 
    
diffs_as_array = np.array(diffs)

rms_diffs = rms(diffs_as_array)
rms_test_sgnls = rms(test_signals)

diffs_rms = []

for diff in diffs:
    diffs_rms.append(rms(diff))

plt.figure(figsize=(15, 7))
plt.plot(20 * np.log10(diffs_rms))
plt.xlabel("Номер сигнала")
plt.ylabel("RMS, dB")
plt.title("RMS от ошибки каждого тестового сигнала")
plt.show()

nmse = rms_diffs / rms_test_sgnls
nmse_db = 20 * np.log10(nmse)


#%%

# Number of signal for review
sig = 7

diffXbetY = abs(test_signals[sig] - decoded_signals[sig])

plot_signals_with_linestyle([test_signals[sig], decoded_signals[sig], diffXbetY], 
                            ['r','b', 'orange'], 
                            ['вход', 'выход', 'разница'], 
                            ["-","-","--"], 
                            title="Разница между тестовым и декодированным сигналом (сигнал #" + str(sig) + ")")

plt.figure(figsize=(15, 7))
plt.stackplot(range(signal_len), diffXbetY, color="orange", labels=["Разница"])
plt.title("Разница между тестовым и декодированным сигналом (сигнал #" + str(sig) + ")")
plt.legend()
plt.show() 


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

#%%
# with open("meanRms.txt", "a") as myfile:
#     myfile.writelines(str(meanRms) + "\n")


#%%
# Импортирование декодированных данных в mat файл для Андрюхи
test_inputs = load_from_file_with_index(files_path, [1, 2, 3, 4, 5, 6], False)
encoded_inputs = []
for person in range(len(test_inputs)):
    encoded_inputs.append([])
    for classe in range(len(test_inputs[person])):
        personClassExp = np.array(test_inputs[person][classe], dtype=np.float64)
        encoded_input = encoder.predict(personClassExp, batch_size=256)
        decoded_input = decoder.predict(encoded_input, batch_size=256)
        encoded_inputs[person].append(decoded_input)
        
scipy.io.savemat('out.mat', mdict={'decoded': encoded_inputs})

#%%
# Импортирование оригинальных данных в mat файл для Андрюхи
test_inputs = load_from_file_with_index(files_path, [1, 2, 3, 4, 5, 6], False)
input_inputs = []
for person in range(len(test_inputs)):
    input_inputs.append([])
    for classe in range(len(test_inputs[person])):
        personClassExp = np.array(test_inputs[person][classe], dtype=np.float64)
        input_inputs[person].append(personClassExp)
        
plt.figure(figsize=(12, 6))
plt.plot(test_inputs[0][0][0], label="До FIR")
plt.plot(input_inputs[0][0][0], label="после FIR")
plt.legend()
plt.show()

scipy.io.savemat('out_filt.mat', mdict={'input': input_inputs})

#%%

# Импортирование оригинальных отфильтрованных данных в mat файл для Андрюхи
test_inputs = load_from_file_with_index(files_path, [1, 2, 3, 4, 5, 6], False)
input_inputs = []
for person in range(len(test_inputs)):
    input_inputs.append([])
    for classe in range(len(test_inputs[person])):
        personClassExp = np.array(test_inputs[person][classe], dtype=np.float64)
        filteredSignal = get_filtered_signals(personClassExp)
        input_inputs[person].append(filteredSignal)
        
plt.figure(figsize=(12, 6))
plt.plot(test_inputs[0][0][0], label="До FIR")
plt.plot(input_inputs[0][0][0], label="после FIR")
plt.legend()
plt.show()

scipy.io.savemat('out_filt.mat', mdict={'filr': input_inputs})
        
# Берем от исходных сигналов преобразование фурье -> подаём на вход автоэнкодера, потом от результата делаем ОПФ
# Размер спектра - 256.
