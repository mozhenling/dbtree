# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:41:06 2022

@author: mozhenling
"""
import numpy as np
from scipy.signal import hilbert
###############################################################################
#------------------------------------------------------------------------------
#-------------- SE, SSES, and others
#------------------------------------------------------------------------------

def sig_real_to_env(sig_real):
    """
    obtain the envelope module of the real signal [1],[2]
    """
    sig_real -= np.mean(sig_real)
    
    if np.iscomplexobj(sig_real): # just in case if it is complex valued
        return np.abs(sig_real)
    else:
        return np.abs(hilbert(sig_real)) # sig_real + j * Hilbert(sig_real)

def sig_real_to_se(sig_real):
    """
    obtain the squared envelope (SE) of the real signal [1],[2]
    """
    return sig_real_to_env(sig_real) ** 2

def sig_real_to_ses(sig_real, DC = 0):
    """
    obtain  the squared envelope spectrum [1],[2]
    Note that the SSES (the square of SES) sometimes is also called squared envelope spectrum (SES) 
    for simplicity see ref.[3]
    """ 
    se = sig_real_to_se(sig_real)
    #-- we remove the DC here before Fourier transform
    se = se - np.mean(se) if DC == 0 else se
    return np.abs(np.fft.fft(se)) / len(se)

def sig_real_to_sses(sig_real):
    """
    obtain the square of the squared envelope 
    spectrum (SSES) of the real signal [1],[2]
    Note that the SSES sometimes is also called squared envelope spectrum (SES) 
    for simplicity see ref.[3]
    """
    return sig_real_to_ses(sig_real)**2

###############################################################################
#-- References
"""
[1] Wang, Dong. "Some further thoughts about spectral kurtosis, spectral L2/L1 norm, 
    spectral smoothness index and spectral Gini index for characterizing repetitive 
    transients." Mechanical Systems and Signal Processing 108 (2018): 360-368.   
[2] Wang, Dong. "Spectral L2/L1 norm: A new perspective for spectral kurtosis 
    for characterizing non-stationary signals." Mechanical Systems and Signal 
    Processing 104 (2018): 290-293.
[3] Borghesani, Pietro, Paolo Pennacchi, and Steven Chatterton. "The relationship 
    between kurtosis-and envelope-based indexes for the diagnostic of rolling
    element bearings." Mechanical Systems and Signal Processing 43.1-2 (2014): 25-43.
"""