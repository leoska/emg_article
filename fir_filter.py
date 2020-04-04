# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:48:02 2020

@author: Leo77
"""

import numpy as np
from scipy import signal
import scipy.io
import matplotlib.pyplot as plt

mat_path = 'D:/emg_article/filt_50Hz_new.mat'

mat = scipy.io.loadmat(mat_path)
h = mat["filt_50Hz"][0]

def get_filtered_signals(signals):
    result = []
    for sgnl in signals:
        result.append( np.convolve(h, sgnl, mode='same') )
    
    return np.array(result)

def plot_filter():
    w,H = signal.freqz(h)
    
    plt.figure(figsize=(15, 7))
    plt.plot(w,abs(H))
    plt.show()
    
def spectral_form_signal(sgnl):
    frex, Pxx = signal.welch(sgnl, nfft = len(sgnl), fs=256)
    iPxx = [ 10 * np.log10(Px) for Px in Pxx ]
    plt.figure(figsize=(15, 7))
    plt.plot(frex, iPxx)
    plt.grid(True)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD')
    plt.legend()
    plt.show()