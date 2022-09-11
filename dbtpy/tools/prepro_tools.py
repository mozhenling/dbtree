# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:30:53 2021

@author: mozhenling
"""
import numpy as np
from librosa import lpc
from scipy.signal import convolve
# from numpy.polynomial import Chebyshev

def detrend(data, n = 1):
    """
    detrend by polynomail of degree 1
    
    Fit a polynomial p(x) = p[0] * x**deg + ... + p[deg] of degree deg to
    points (x, y). Returns a vector of coefficients p that minimises the
    squared error in the order deg, deg-1, … 0
    @author: mozhenling
    """
    x = np.arange(0, len(data))
    # Detrend with a 2d order polynomial
    model = np.polyfit(x, data, n )
    predicted = np.polyval(model, x)
    return data - predicted

def pre_whiten(x, Na = 100):
    """
    #-- pre-whitening
    #-- Pre-whitening of the signal by AR filtering (optional)
    #   (always helpful in detection problems)
    #-- you need pip install scipy and librosa
    # @author: mozhenling
    """
    x = x - np.mean(x)
     # Order of the linear filter
    #-- Linear Prediction Coefficients via Burg’s method
    a = lpc(x,Na) 
    x = convolve(a,x) # equivalence to fftfilt in matlab
    x = x[Na:-100]
    return x
    #--------------------------------------------------------------------------

def bearing_fault_freq(f_shaft_rpm =1, D = 38.5, d = 7.938,  gama = 0, n = 9):
    """
    calculate fault frequencies based on bearing geometry and shaft speed
    in Hz ( rpm / 60 )
    
    inputs:
        -D     # the pitch  diameter
        -d     # the  rolling  element diameter
        -fr    # the shaft rotation frequency in rpm
        -gama  # the contact angle
        -n     # the number of rolling elements
        
    outputs:
        NA
    ref.:
        Wang, Mingfang, et al. "Harmonic L2/L1 norm for bearing fault 
        diagnosis." IEEE Access 7 (2019): 27313-27321.
    @author: mozhenling 
    """
    f_shaft_Hz = f_shaft_rpm / 60
    f_dict = {}
    f_dict['f_shaft'] = f_shaft_Hz
    f_dict['f_inner'] = 0.5 * n * f_shaft_Hz * (1 + (d / D) * np.cos(gama) )
    f_dict['f_ball'] = ( 0.5 * D * f_shaft_Hz / d ) * (1 - ( (d / D) * np.cos(gama) )**2 )
    f_dict['f_cage'] =  0.5 * f_shaft_Hz * (1 - (d / D) * np.cos(gama)) 
    f_dict['f_outer'] = 0.5 * n * f_shaft_Hz * (1 - (d / D) * np.cos(gama))
     
    return f_dict

def planetary_gearbox_freq(f_shaft_rpm, Zr, Zp, Zs, K ):
        """
        set the fault-related frequencies
        
        input:
            #-- the number of teeth of 
            #- ring gear, planetary gear, sun gear,
            # and the number of planetary, e.g.
            Zr, Zp, Zs, K = 74, 23, 25, 3
            
        use the cal_freq() in tools package to get the frequencies
        
        ref.:
            Lei Y, Lin J, Zuo MJ, et al. Condition monitoring and
            fault diagnosis of planetary gearboxes: a review.
            Measurement 2014; 48: 292–305.
        
        """
        #-- the number of teeth of 
        #- ring gear, planetary gear, sun gear,
        # and the number of planetary
        # Zr, Zp, Zs, K = 74, 23, 25, 3
        
        f_shaft_Hz = f_shaft_rpm / 60
        f_dict = {}
        
        f_dict['f_shaft'] = f_shaft_Hz                      # shaft frequency 
        f_dict['f_sun_r'] = f_shaft_Hz                      # rotating frequence of sun gear 
        f_dict['f_mesh'] = f_shaft_Hz * Zr*Zs / (Zr+Zs)     # meshing frequency
        f_dict['f_sun_d'] = f_shaft_Hz * Zr / (Zr+Zs)       # distributed fault frequency of sun gear
        f_dict['f_sun_l'] = K * f_dict['f_sun_d']               # localized fault frequency of sun gear
        f_dict['f_plan'] = f_shaft_Hz * Zr * Zs / (Zp * (Zr+Zs)) # fault frequency of plenatery gear
        f_dict['f_ring_d'] = f_shaft_Hz * Zs / (Zr+Zs)      # distributed fault frequency of ring gear
        f_dict['f_ring_l'] = K * f_dict['f_ring_d']             # localized fault frequency of ring gear

        return f_dict

if __name__ == '__main__':
    # a = cal_freq()
    a = 1
