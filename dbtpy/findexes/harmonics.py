# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:58:55 2022

@author: mozhenling
"""
import numpy as np
from dbtpy.findexes.pq_norms import pq_norm
from scipy.stats import kurtosis
###############################################################################
#--------------fault findexes dependent of fault frequencies
###############################################################################
def harEstimation(seq, f_target, harN = 3, fs = 10e3, dev1 = 0.025, 
                  sig_len_original = None, fre2seq = None):
    '''
        inputs:
            -seq                    # half demodulation spectrum 
            -harN = 5               # number of considered harmonics
            -fs = 10e3              # sampling frequency (Hz)
            -f_target = 62.2631     # the target fault characteristic frequency (Hz)
            -dev1 = 0.05            # deviation percentage for estimating a local harmonic
            -dev2 = 0.5             # deviation percentage for estimating a fault harmonic to noise ratio
            -fre2seq: convert frequency to sequency index
        output:
            - amplitude and index estimation of fault characteristic frequency in sequence

        ref.:
            Mo, Zhenling, et al. "Weighted cyclic harmonic-to-noise ratio for rolling
            element bearing fault diagnosis." IEEE Transactions on Instrumentation and
            Measurement 69.2 (2019): 432-442.

        '''
    # ------- initialization
    fHarAmp = np.zeros([harN, 1])  # the amplitudes of the fault frequency harmonics
    fHarInd = np.zeros([harN, 1])  # the findexes of the fault frequency harmonics
    seq = seq.squeeze()
    #--fre2seq: convert frequency to sequency index
    
    if  fre2seq is not None:
        fre2seq =  fre2seq
    elif sig_len_original is not None:
        fre2seq = sig_len_original / fs
    else:
        fre2seq = len(seq) / fs

    fHar1Seq = f_target * fre2seq  # f_target in Hz to f_target in seq
    fHarInd_temp = fHar1Seq  # harmonic index in seq
    delta1 = int(np.round(fHar1Seq * dev1))

    for i in range(harN):
        # -- find the real position of each target harmonic
        f_lw = int(np.floor(fHarInd[0] + fHarInd_temp - delta1))
        f_up = int(np.ceil(fHarInd[0] + fHarInd_temp + delta1))
        # - get a sliced range of seq.
        seq_est = seq[f_lw:f_up]
        seq_len = len(seq_est)
        if seq_len > 1:
            [fHarAmp[i], fmaxIndex] = [np.max(seq_est), np.argmax(seq_est)]
            
        elif seq_len == 1:
            (fHarAmp[i], fmaxIndex) =(seq[f_lw ], 0) if seq[f_lw ]> seq[f_up] else (seq[f_up], 1)
            
        else:
            (fHarAmp[i], fmaxIndex) = (seq[f_lw], 0)
            
        fHarInd_temp = f_lw + fmaxIndex  
        fHarInd[i] = fHarInd_temp  # store the estimiated position

    return fHarAmp, fHarInd

def vanillaSNR(seq, f_target, harN = 3, fs = 10e3, dev1 = 0.025, dev2 = 0.5, 
               sig_len_original = None, fre2seq = None):
    # prevent zero denominator
    inf_preventor = np.spacing(1)
    fHarAmp, fHarInd = harEstimation(seq, f_target, harN , fs, dev1 ,
                                     sig_len_original,fre2seq)
    delta2 = int(np.round(fHarInd[0] * dev2))
    noise = sum(seq[int(fHarInd[0])-delta2 : int(fHarInd[-1]) + delta2]) - sum(fHarAmp)
    return 10 * np.log10( sum(fHarAmp) / (noise + inf_preventor) ).squeeze()

def harL2L1norm(seq, f_target, harN = 3, fs = 10e3, dev1 = 0.025, dev2 = 0.5,
                sig_len_original = None, fre2seq = None):
    _, fHarInd = harEstimation(seq, f_target, harN , fs, dev1 , 
                               sig_len_original,fre2seq)
    delta2 = int(np.round(fHarInd[0] * dev2))
    har_pq = np.zeros([harN, 1])
    for i in range(harN):
        har_seq = seq[int(fHarInd[i])-delta2 : int(fHarInd[i]) + delta2]
        har_pq[i]=pq_norm(har_seq, p=2, q=1)
    return np.mean(har_pq)

def harkurtosis(seq, f_target, harN = 3, fs = 10e3, dev1 = 0.025, dev2 = 0.5,
                sig_len_original = None,fre2seq = None):
    _, fHarInd = harEstimation(seq, f_target, harN , fs, dev1 ,
                               sig_len_original,fre2seq)
    delta2 = int(np.round(fHarInd[0] * dev2))
    har_pq = np.zeros([harN, 1])
    for i in range(harN):
        har_seq = seq[int(fHarInd[i])-delta2 : int(fHarInd[i]) + delta2]
        har_pq[i]=kurtosis(har_seq)
    return np.mean(har_pq)

#------------------------------------------------------------------------------
#--------------cyclic to harmonic ratio (CHNR)
#------------------------------------------------------------------------------
def CHNR(seq, f_target, harN = 3, fs = 10e3, dev1 = 0.025, dev2 = 0.5, 
         sig_len_original = None,fre2seq = None):
    '''
    The cycic harmonic to noise ratio without the white noise threshold
    inputs:
        -seq                    # the real-valued signal
        -harN = 5               # number of considered harmonics
        -fs = 10e3              # sampling frequency (Hz)
        -f_target = 62.2631     # the target fault characteristic frequency (Hz)
        -dev1 = 0.025           # half of the deviation percentage for estimating a local harmonic
        -dev2 = 0.5             # deviation percentage for estimating a fault harmonic to noise ratio
        -
    output:
        -the CHNR of the target frequency
    ref.:
        Mo, Zhenling, et al. "Weighted cyclic harmonic-to-noise ratio for rolling 
        element bearing fault diagnosis." IEEE Transactions on Instrumentation and
        Measurement 69.2 (2019): 432-442.
         
    '''
    inf_preventor = np.spacing(1)   # prevent zero denominator
    fHarAmp, fHarInd = harEstimation(seq, f_target, harN , fs, dev1 ,
                                     sig_len_original,fre2seq)
    delta1 = int(np.round(fHarInd[0] * dev1))
    kN = int(np.round((dev2 - dev1) / (2 * dev1)))  # the number of considered sub-bands around the target harmonic
    fHarUp = np.zeros([kN, harN])   # upper neighboring local maxima of the target harmonic
    fHarLow = np.zeros([kN, harN])  # lower neighboring local maxima of the target harmonic
    chnr = np.zeros([harN, 1])      # the cyclic harmonic to noise ratio

    #-- conisder the neighboring local maxima, i.e., sampling of local maxima
    for i, (h_amp, h_ind) in enumerate(zip(fHarAmp, fHarInd)):
        for k in range(kN):
            # k= 0, 1, 2, ..., kN
           fHarUp[k,i] = np.max(seq[int(np.floor( h_ind + (2*k + 1) * delta1)): int(np.ceil( h_ind + (2*k + 3) * delta1))])
           fHarLow[k,i] = np.max(seq[int(np.floor( h_ind + (-2*k - 3) * delta1)): int(np.ceil( h_ind + (-2*k -1) * delta1))])
        #-- the cycic harmonic to noise ratio in percentage
        chnr[i] = h_amp / ( (h_amp + sum(fHarUp[:,i]) + sum(fHarLow[:,i])) + inf_preventor )
    #-- other option
    # CHNR[i] = 20 * np.log10( h_amp / (sum(fHarUp[:,i]) + sum(fHarLow[:,i])) + inf_preventor)  )
    return np.mean(chnr)  
 