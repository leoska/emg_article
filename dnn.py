# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:43:14 2020

@author: Leo77
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from random import shuffle
from sklearn import metrics
from tensorflow.keras import utils
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Activation
from tensorflow.keras.models import Sequential, Model

#%%
tf.keras.backend.clear_session()

#%%
def load_dataset(P_all, persons):
    features = []
    labels = []
    personId = 0
    classId = 0
    testId = 0
    features_2d = []
    labels_2d = []

    for subject_id in persons:
        features.append([])
        labels.append([])
        subject_obj = P_all[subject_id - 1]
        for subject in subject_obj:
            for class_obj in subject:
                features[personId].append([])
                labels[personId].append([])
                for class_n in class_obj:
                    for test_number in np.array(class_n).transpose():
                        features[personId][classId].append(test_number)
                        
                        labels_tmp = np.zeros(9, dtype=np.int32)
                        labels_tmp[classId] = 1
                        labels[personId][classId].append(labels_tmp)
                        
                        features_2d.append(test_number)
                        labels_2d.append(labels_tmp)
                classId += 1
        personId += 1
        classId = 0
        
    features_2d = np.asarray(features_2d)
    labels_2d = np.asarray(labels_2d)
    
    c = list(zip(features_2d, labels_2d))
    shuffle(c)
    features_2d, labels_2d = zip(*c)
    
    features_2d = np.asarray(features_2d, dtype=np.float64)
    labels_2d = np.asarray(labels_2d, dtype=np.int64)

    return (features, labels, features_2d, labels_2d)

def show_confusion_matrix(validations, predictions):
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                #xticklabels=LABELS,
                #yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
    
# Функция, которая выводит метрики обучения нейронной сети
def plot_history(history):
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.title("График ошибки на тренировочных данных")
    plt.xlabel("Эпоха #")
    plt.ylabel("Значение ошибки")
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history["val_loss"])
    plt.title("График ошибки на проверочных данных")
    plt.xlabel("Эпоха #")
    plt.ylabel("Значение ошибки")
    plt.show()
    
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    x = [acc * 100 for acc in history.history["acc"]]
    plt.plot(np.round(x, 2))
    plt.title("Доля верных ответов на тренировочных данных")
    plt.xlabel("Эпоха #")
    plt.ylabel("Процент верных ответов")
    ax = plt.subplot(1, 2, 2)
    x = [val_acc * 100 for val_acc in history.history["val_acc"]]
    plt.plot(np.round(x, 2))
    plt.title("Доля верных ответов на проверочных данных")
    plt.xlabel("Эпоха #")
    plt.ylabel("Процент верных ответов")
    plt.show()


#%%
# Загружаем датасет
mat_path = 'all_future_for_lenia_real.mat'

# Читаем файл матлаба
mat = scipy.io.loadmat(mat_path)

# Достаём данные из файла и начинаем их парсить
P_all = mat['P_all']

# 6 persons ( 1 - Андрей, 2 - АндрСеме, 3 - Лёня, 4 - Миша, 5 - Юра, 6 - Алексей Олегович )
train_persons = [1, 2, 4, 5]
test_persons = [3, 6]

# Длина вектора признаков
feature_len = 29

classes_count = 9

#%%

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

#%%

# Load dataset from directory files_path
# Создание датасета (обучение и тестовые) из файлов в директории files_path
(train_features, train_labels, train_f_2d, train_l_2d) = load_dataset(P_all, train_persons)
(test_features, test_labels, test_f_2d, test_l_2d) = load_dataset(P_all, test_persons)

    
#%%
# Model of DeepClassificator

deep_model = Sequential()
deep_model.add(Input(shape=(feature_len,)))
deep_model.add(Dense(80, input_shape=(feature_len,), activation='sigmoid')) #11
#deep_model.add(Dense(25, input_shape=(50,), activation='sigmoid')) #5
#deep_model.add(Dense(, input_shape=(5,), activation='sigmoid')) #2
deep_model.add(Dense(classes_count, input_shape=(500,), activation='softmax'))

deep_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ["accuracy"])

print(deep_model.summary())

#%%

# Обучаем нейросетку
history = deep_model.fit(train_f_2d, train_l_2d,
                      epochs=50,
                      batch_size=64,
                      # shuffle=True,
                      validation_split=0.2,
                      verbose=1
)


plot_history(history)

#%%

# Тестирование
test_dataset = tf.data.Dataset.from_tensor_slices((test_f_2d, test_l_2d))
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(64)

loss, accuracy = deep_model.evaluate(test_dataset, verbose = 1) # evaluating model on test data

loss = float("{0:.3f}".format(loss))
accuracy = float("{0:.3f}".format(accuracy))

print('Доля верных ответов на тестовых данных, в процентах: ' + str(round(accuracy * 100, 4)))

#%%

# Предугадывание

print("\n--- Confusion matrix for test data ---\n")

test_set = np.copy(test_f_2d)

np.random.shuffle(test_set)

y_pred_test = deep_model.predict(test_set)
# Take the class with the highest probability from the test predictions
y_test = np.asarray(y_pred_test).reshape((len(y_pred_test), np.prod(np.asarray(y_pred_test).shape[1:])))
max_y_pred_test = np.argmax(y_test, axis=1)
max_y_test = np.argmax(test_l_2d, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

