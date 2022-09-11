# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 17:26:39 2020

@author: mozhenling
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift, hilbert

def indexes(fs, data, flag = 'half'):
    Ns = len(data)
    n = np.arange(Ns)  # start = 0, stop = Ns -1, step = 1
    t = n / fs  # time index
    fn = n * fs / Ns -0.5*fs # frequency index, Fs / Ns = frequency resoluton
    Amp_fn = abs(np.fft.fftshift(np.fft.fft(data))) /Ns
    if flag == 'half': 
        return t, fn[len(fn)//2:], Amp_fn[len(fn)//2:]
    elif flag == 'full':
        return t, fn, Amp_fn
    else:
        return t, fn[len(fn)//2:], Amp_fn[len(fn)//2:]

def tf_indexes(fs, data, flag = 'half'):
    Ns = len(data)
    n = np.arange(Ns)  # start = 0, stop = Ns -1, step = 1
    t = n / fs  # time index
    fn = n * fs / Ns -0.5*fs # frequency index, Fs / Ns = frequency resoluton
    Amp_fn = abs(np.fft.fftshift(np.fft.fft(data))) /Ns
    if flag == 'half': 
        return t, fn[len(fn)//2:], Amp_fn[len(fn)//2:]
    elif flag == 'full':
        return t, fn, Amp_fn
    else:
        return t, fn[len(fn)//2:], Amp_fn[len(fn)//2:]

def tf_t_show(fs, data):
    '''
    show time domain waveform only
    '''
    plt.figure()
    Ns = len(data)
    n = np.arange(Ns)  # start = 0, stop = Ns -1, step = 1
    t = n / fs  # time index
    plt.plot(t, data, 'b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude ')
    plt.title('Time Domain')
    plt.show()

def tf_f_show(fs, data, flag = 'half'):
    """
    show frequency power spectrum only
    flag = half, 0 to fs // 2
    flag = full, -fs//2 to fs //2
    """
    plt.figure()
    Ns = len(data)
    n = np.arange(Ns)  # start = 0, stop = Ns -1, step = 1
    t = n / fs  # time index
    fn = n * fs / Ns -0.5*fs  # frequency index, Fs / Ns = frequency resoluton
    Amp_fn = 2 * abs(np.fft.fftshift(np.fft.fft(data))) /Ns
    if flag == 'half':
        fn = n * fs / Ns  # frequency index, Fs / Ns = frequency resoluton
        plt.plot(fn[len(fn)//2:], Amp_fn[len(fn)//2:], 'b')
    elif flag == 'full':
        plt.plot(fn, Amp_fn, 'b')
    else:
        raise Exception('Please set the flag = \'half\' or flag = \'full\'')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude ')
    plt.title('Frequency Domain')
    plt.show()

def tf_two_show(fs, data):
    '''
    show both time and frequency domains in separate figures
    '''
    t, fn, Amp_fn = indexes(fs, data) 
    #-- time domain
    plt.figure()
    plt.plot(t, data, 'b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude ')
    plt.title('Time Domain')
    plt.show()
    #-- frequency domain  
    plt.figure()
    plt.plot(fn, Amp_fn, 'g')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude ')
    plt.title('Frequency Domain')
    plt.show()
    

def tf_one_show(fs, data, flag = 'half'):
    """
    show both time and frequency domains in one figure
    """
    t, fn, Amp_fn = indexes(fs, data, flag) 
    #-- time domain
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, data, 'b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude ')
    plt.title('Time Domain [Above], Frequency Domain [Below]')
    #-- frequency domain     
    plt.subplot(2, 1, 2)
    plt.plot(fn, Amp_fn, 'g')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude ')
    plt.show()
    #plt.title('Raw Signal [Frequency Domain]')    

def tf_2datas_one_show(fs, data_1, data_2, names = ['data_1',"data_2"] ):
    """
    show both time and frequency domains in one figure
    """
    t, fn, Amp_fn_1 = indexes(fs, data_1) 
    _, _, Amp_fn_2 = indexes(fs, data_2) 
    #-- time domain
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, data_1, 'b')
    plt.plot(t, data_2, 'r')
    plt.legend(names)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude ')
    plt.title('Time Domain [Above], Frequency Domain [Below]')
    #-- frequency domain     
    plt.subplot(2, 1, 2)
    plt.plot(fn, Amp_fn_1, 'b')
    plt.plot(fn, Amp_fn_2, 'r')
    plt.legend(names)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude ')
    plt.show()
    #plt.title('Raw Signal [Frequency Domain]')  
    
def tf_Ndatas_one_show(fs, data_list = [], names = [] ):
    """
    show both time and frequency domains of N datas in one figure 
    """
    Amp_fn = []
    for i in range(len(data_list)):
        t, fn, Amp = indexes(fs, data_list[i]) 
        Amp_fn.append(Amp)
    #-- time domain
    plt.figure()
    plt.subplot(2, 1, 1)
    for i in range(len(data_list)): 
        plt.plot(t, data_list[i])
    plt.legend(names)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude ')
    plt.title('Time Domain [Above], Frequency Domain [Below]')
    #-- frequency domain     
    plt.subplot(2, 1, 2)
    for i in range(len(data_list)):    
        plt.plot(fn, Amp_fn[i])
    plt.legend(names)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude ')
    plt.show()
    #plt.title('Raw Signal [Frequency Domain]')  
    
def tf_Ndatas_withAxis_f_show(Axis, data_list = [], names = []):
    """
    show  frequency domains of N datas in one figure 
    """
    plt.figure()
    for i in range(len(data_list)): 
        plt.plot(Axis, data_list[i], label=names[i])
    plt.xlabel('frequency (Hz)')
    plt.ylabel('Amplitude ')
    plt.legend(loc=0,ncol=1)
    # plt.title('')
 

def tf_mul_show(fs, data_1, data_2):
    """
    show the freqeuncy domain of data_1 times frequency domain of data_2
    """
    _, fn, Amp_fn_1 = indexes(fs, data_1) 
    _, fn, Amp_fn_2 = indexes(fs, data_2) 
    Amp_fn = Amp_fn_1 * Amp_fn_2
    #-- time domain
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(fn, Amp_fn_1, 'b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude ')
    
    plt.subplot(3, 1, 2)
    plt.plot(fn, Amp_fn_2, 'b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude ')
    # plt.title('Time Domain [Above], Frequency Domain [Below]')
    #-- frequency domain     
    plt.subplot(3, 1, 3)
    plt.plot(fn, Amp_fn, 'g')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude ')
    #plt.title('Raw Signal [Frequency Domain]')   

def tf_sq_show(fs, data):
    """
    show squared envelope and its spectrum of data
    """
    se = abs(hilbert(data))**2   
    se = se - np.mean(se)
    t, fn, Amp_fn = indexes(fs, se) 
    #-- time domain
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, data, 'b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude ')
    plt.title('Squared Envelope Time  [Above], Frequency [Below]')
    #-- frequency domain     
    plt.subplot(2, 1, 2)
    plt.plot(fn, Amp_fn, 'g')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude ')
    plt.show()
    #plt.title('Raw Signal [Frequency Domain]')
    
def detrend(data, n = 2):
    """
    detrend by polynomail of degree 2
    """
    x = np.arange(0, len(data))
    # Detrend with a 2d order polynomial
    model = np.polyfit(x, data, n )
    predicted = np.polyval(model, x)
    return data - predicted

