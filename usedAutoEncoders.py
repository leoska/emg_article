# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 00:02:18 2020

@author: leonidkotov
"""

from tensorflow.keras import utils
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l1

# 200 epochs
def deepautoencoder_one(signal_len):
    lambda_l1 = 10e-5
    
    encoder = Sequential()
    encoder.add(Input(shape=(signal_len,)))
    encoder.add(Dense(256, input_shape=(signal_len,)))
    encoder.add(Activation("linear"))
    encoder.add(Dense(128, input_shape=(256,)))
    encoder.add(Activation("linear"))
    encoder.add(Dense(64, input_shape=(128,)))
    encoder.add(Activation("linear"))
    encoder.add(Dense(32, input_shape=(64,)))
    encoder.add(Activation("linear"))
    encoder.add(Dense(16, input_shape=(32,)))
    encoder.add(Activation("linear"))
    encoder.add(Dense(8, input_shape=(16,), activity_regularizer=l1(lambda_l1)))
    encoder.add(Activation("linear"))
    
    decoder = Sequential()
    decoder.add(Input(shape=(8,)))
    decoder.add(Dense(16, input_shape=(8,)))
    decoder.add(Activation("linear"))
    decoder.add(Dense(32, input_shape=(16,)))
    decoder.add(Activation("linear"))
    decoder.add(Dense(64, input_shape=(32,)))
    decoder.add(Activation("linear"))
    decoder.add(Dense(128, input_shape=(64,)))
    decoder.add(Activation("linear"))
    decoder.add(Dense(256, input_shape=(128,)))
    decoder.add(Activation("linear"))
    decoder.add(Dense(signal_len, input_shape=(256,)))
    decoder.add(Activation("linear"))
    
    input_signal = Input(shape=(signal_len,))
    ec_out = encoder(input_signal)
    dc_out = decoder(ec_out)
    
    autoencoder = Model(inputs = input_signal, outputs = dc_out)
    
    return autoencoder, encoder, decoder

# 200 epochs
def deepautoencoder_two(signal_len):
    lambda_l1 = 10e-5
    
    encoder = Sequential()
    encoder.add(Input(shape=(signal_len,)))
    encoder.add(Dense(256, input_shape=(signal_len,)))
    encoder.add(Activation("linear"))
    encoder.add(Dense(128, input_shape=(256,)))
    encoder.add(Activation("linear"))
    encoder.add(Dense(64, input_shape=(128,)))
    encoder.add(Activation("linear"))
    encoder.add(Dense(32, input_shape=(64,)))
    encoder.add(Activation("linear"))
    encoder.add(Dense(16, input_shape=(32,), activity_regularizer=l1(lambda_l1)))
    encoder.add(Activation("linear"))
    
    decoder = Sequential()
    decoder.add(Input(shape=(16,)))
    decoder.add(Dense(32, input_shape=(16,)))
    decoder.add(Activation("linear"))
    decoder.add(Dense(64, input_shape=(32,)))
    decoder.add(Activation("linear"))
    decoder.add(Dense(128, input_shape=(64,)))
    decoder.add(Activation("linear"))
    decoder.add(Dense(256, input_shape=(128,)))
    decoder.add(Activation("linear"))
    decoder.add(Dense(signal_len, input_shape=(256,)))
    decoder.add(Activation("linear"))
    
    input_signal = Input(shape=(signal_len,))
    ec_out = encoder(input_signal)
    dc_out = decoder(ec_out)
    
    autoencoder = Model(inputs = input_signal, outputs = dc_out)
    
    return autoencoder, encoder, decoder

# 200 epochs
def deepautoencoder_three(signal_len):
    lambda_l1 = 10e-5
    
    encoder = Sequential()
    encoder.add(Input(shape=(signal_len,)))
    encoder.add(Dense(256, input_shape=(signal_len,)))
    encoder.add(Activation("linear"))
    encoder.add(Dense(128, input_shape=(256,)))
    encoder.add(Activation("linear"))
    encoder.add(Dense(64, input_shape=(128,)))
    encoder.add(Activation("linear"))
    encoder.add(Dense(32, input_shape=(64,), activity_regularizer=l1(lambda_l1)))
    encoder.add(Activation("linear"))
    
    decoder = Sequential()
    decoder.add(Input(shape=(32,)))
    decoder.add(Dense(64, input_shape=(32,)))
    decoder.add(Activation("linear"))
    decoder.add(Dense(128, input_shape=(64,)))
    decoder.add(Activation("linear"))
    decoder.add(Dense(256, input_shape=(128,)))
    decoder.add(Activation("linear"))
    decoder.add(Dense(signal_len, input_shape=(256,)))
    decoder.add(Activation("linear"))
    
    input_signal = Input(shape=(signal_len,))
    ec_out = encoder(input_signal)
    dc_out = decoder(ec_out)
    
    autoencoder = Model(inputs = input_signal, outputs = dc_out)
    
    return autoencoder, encoder, decoder

# 200 epochs
def deepautoencoder_four(signal_len):
    lambda_l1 = 10e-5
    
    encoder = Sequential()
    encoder.add(Input(shape=(signal_len,)))
    encoder.add(Dense(256, input_shape=(signal_len,)))
    encoder.add(Activation("linear"))
    encoder.add(Dense(128, input_shape=(256,)))
    encoder.add(Activation("linear"))
    encoder.add(Dense(64, input_shape=(128,), activity_regularizer=l1(lambda_l1)))
    encoder.add(Activation("linear"))
    
    decoder = Sequential()
    decoder.add(Input(shape=(64,)))
    decoder.add(Dense(128, input_shape=(64,)))
    decoder.add(Activation("linear"))
    decoder.add(Dense(256, input_shape=(128,)))
    decoder.add(Activation("linear"))
    decoder.add(Dense(signal_len, input_shape=(256,)))
    decoder.add(Activation("linear"))
    
    input_signal = Input(shape=(signal_len,))
    ec_out = encoder(input_signal)
    dc_out = decoder(ec_out)
    
    autoencoder = Model(inputs = input_signal, outputs = dc_out)
    
    return autoencoder, encoder, decoder

# 200 epochs
def deepautoencoder_five(signal_len):
    lambda_l1 = 10e-5
    
    encoder = Sequential()
    encoder.add(Input(shape=(signal_len,)))
    encoder.add(Dense(256, input_shape=(signal_len,)))
    encoder.add(LeakyReLU(alpha=0.2))
    encoder.add(Dense(128, input_shape=(256,)))
    encoder.add(LeakyReLU(alpha=0.2))
    encoder.add(Dense(64, input_shape=(128,), activity_regularizer=l1(lambda_l1)))
    encoder.add(Activation("linear"))
    
    decoder = Sequential()
    decoder.add(Input(shape=(64,)))
    decoder.add(Dense(128, input_shape=(64,)))
    decoder.add(LeakyReLU(alpha=0.2))
    decoder.add(Dense(256, input_shape=(128,)))
    decoder.add(LeakyReLU(alpha=0.2))
    decoder.add(Dense(signal_len, input_shape=(256,), activation='tanh'))
    
    input_signal = Input(shape=(signal_len,))
    ec_out = encoder(input_signal)
    dc_out = decoder(ec_out)
    
    autoencoder = Model(inputs = input_signal, outputs = dc_out)
    
    return autoencoder, encoder, decoder

# 200 epochs
def deepautoencoder_six(signal_len):
    lambda_l1 = 10e-5
    
    encoder = Sequential()
    encoder.add(Input(shape=(signal_len,)))
    encoder.add(Dense(256, input_shape=(signal_len,)))
    encoder.add(LeakyReLU(alpha=0.2))
    encoder.add(Dense(128, input_shape=(256,)))
    encoder.add(LeakyReLU(alpha=0.2))
    encoder.add(Dense(64, input_shape=(128,)))
    encoder.add(LeakyReLU(alpha=0.2))
    encoder.add(Dense(32, input_shape=(64,), activity_regularizer=l1(lambda_l1)))
    encoder.add(Activation("linear"))
    
    decoder = Sequential()
    decoder.add(Input(shape=(32,)))
    decoder.add(Dense(64, input_shape=(32,)))
    decoder.add(LeakyReLU(alpha=0.2))
    decoder.add(Dense(128, input_shape=(64,)))
    decoder.add(LeakyReLU(alpha=0.2))
    decoder.add(Dense(256, input_shape=(128,)))
    decoder.add(LeakyReLU(alpha=0.2))
    decoder.add(Dense(signal_len, input_shape=(256,), activation='tanh'))
    
    input_signal = Input(shape=(signal_len,))
    ec_out = encoder(input_signal)
    dc_out = decoder(ec_out)
    
    autoencoder = Model(inputs = input_signal, outputs = dc_out)
    
    return autoencoder, encoder, decoder

def deepautoencoder_seven(signal_len):
    lambda_l1 = 10e-5
    
    encoder = Sequential()
    encoder.add(Input(shape=(signal_len,)))
    encoder.add(Dense(signal_len * 2, input_shape=(signal_len,)))
    encoder.add(LeakyReLU(alpha=0.2))
    encoder.add(Dense(signal_len, input_shape=(signal_len * 2,), activity_regularizer=l1(lambda_l1)))
    encoder.add(Activation("linear"))
    
    decoder = Sequential()
    decoder.add(Input(shape=(signal_len,)))
    decoder.add(Dense(signal_len * 2, input_shape=(signal_len,)))
    decoder.add(LeakyReLU(alpha=0.2))
    decoder.add(Dense(signal_len, input_shape=(signal_len * 2,), activation='tanh'))
    
    input_signal = Input(shape=(signal_len,))
    ec_out = encoder(input_signal)
    dc_out = decoder(ec_out)
    
    autoencoder = Model(inputs = input_signal, outputs = dc_out)
    
    return autoencoder, encoder, decoder

def deepautoencoder_eight(signal_len):
    lambda_l1 = 10e-5
    
    encoder = Sequential()
    encoder.add(Input(shape=(signal_len,)))
    encoder.add(Dense(signal_len * 2, input_shape=(signal_len,)))
    encoder.add(LeakyReLU(alpha=0.2))
    encoder.add(Dense(signal_len * 3, input_shape=(signal_len * 2,)))
    encoder.add(LeakyReLU(alpha=0.2))
    encoder.add(Dense(signal_len, input_shape=(signal_len * 3,), activity_regularizer=l1(lambda_l1)))
    encoder.add(Activation("linear"))
    
    decoder = Sequential()
    decoder.add(Input(shape=(signal_len,)))
    decoder.add(Dense(signal_len * 2, input_shape=(signal_len,)))
    decoder.add(LeakyReLU(alpha=0.2))
    decoder.add(Dense(signal_len * 3, input_shape=(signal_len * 2,)))
    decoder.add(LeakyReLU(alpha=0.2))
    decoder.add(Dense(signal_len, input_shape=(signal_len * 3,), activation='tanh'))
    
    input_signal = Input(shape=(signal_len,))
    ec_out = encoder(input_signal)
    dc_out = decoder(ec_out)
    
    autoencoder = Model(inputs = input_signal, outputs = dc_out)
    
    return autoencoder, encoder, decoder