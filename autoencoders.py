#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:11:03 2020

@author: leonidkotov
"""

from tensorflow.keras import utils
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Sequential, Model

def simpleautoencoder(signal_len):
    encoder = Sequential()
    encoder.add(Input(shape=(signal_len,)))
    encoder.add(Dense(32, input_shape=(signal_len,)))
    encoder.add(LeakyReLU(alpha=0.2))
    
    decoder = Sequential()
    decoder.add(Input(shape=(32,)))
    decoder.add(Dense(signal_len, input_shape=(32,)))
    decoder.add(LeakyReLU(alpha=0.2))
    
    input_signal = Input(shape=(signal_len,))
    ec_out = encoder(input_signal)
    dc_out = decoder(ec_out)
    
    autoencoder = Model(inputs = input_signal, outputs = dc_out)
    
    return autoencoder, encoder, decoder

def deepautoencoder(signal_len):
    encoder = Sequential()
    encoder.add(Input(shape=(signal_len,)))
    encoder.add(Dense(64, input_shape=(signal_len,), activation="linear"))
    #encoder.add(LeakyReLU(alpha=0.2))
    encoder.add(Dense(32, input_shape=(64,), activation="linear"))
    #encoder.add(LeakyReLU(alpha=0.2))
    
    decoder = Sequential()
    decoder.add(Input(shape=(32,)))
    decoder.add(Dense(64, input_shape=(32,), activation="linear"))
    #decoder.add(LeakyReLU(alpha=0.2))
    decoder.add(Dense(signal_len, input_shape=(64,), activation="linear"))
    #decoder.add(LeakyReLU(alpha=0.2))
    
    input_signal = Input(shape=(signal_len,))
    ec_out = encoder(input_signal)
    dc_out = decoder(ec_out)
    
    autoencoder = Model(inputs = input_signal, outputs = dc_out)
    
    return autoencoder, encoder, decoder

