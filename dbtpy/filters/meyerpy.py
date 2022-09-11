# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:45:32 2018
Empirical Wavelet Transform implementation for 1D signals
Original paper: 
Gilles, J., 2013. Empirical Wavelet Transform. IEEE Transactions on Signal Processing, 61(16), pp.3999–4010. 
Available at: http://ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=6522142.
Original Matlab toolbox: https://www.mathworks.com/matlabcentral/fileexchange/42141-empirical-wavelet-transforms
 
@author: Vinícius Rezende Carvalho
Programa de pós graduação em engenharia elétrica - PPGEE UFMG
Universidade Federal de Minas Gerais - Belo Horizonte, Brazil
Núcleo de Neurociências - NNC 
modified by mozhenling for optimization based fault diagnosis
"""

import numpy as np
#%EWT functions 
def meyer(f, boundaries, filter_num = 3):
    """
     =========================================================================
     ewt,  mfb ,boundaries = ewt1d(f, N = 5, log = 0,detect = "locmax", completion = 0, reg = 'average', lengthFilter = 10,sigmaFilter = 5):
     
     Perform the Empirical Wavelet Transform of f over N scales. See 
     also the documentation of EWT_Boundaries_Detect for more details about
     the available methods and their parameters.
    
     Inputs:
       -f: the 1D input signal 
       -boundariers: the decision variable (without unit like Hz, just the index)

     Outputs:
       -ewt: contains first the low frequency component and
             then the successives frequency subbands
       -mfb: contains the filter bank (in the Fourier domain)
       -boundaries: vector containing the set of boundaries corresponding
                    to the Fourier line segmentation (normalized between
                    0 and Pi)
     Original MATLAB Version:
     Author: Jerome Gilles
     Institution: UCLA - Department of Mathematics
     Year: 2013
     Version: 2.0
     
     Python Version: Vinícius Rezende Carvalho - vrcarva@ufmg.br
     Universidade Federal de Minas Gerais - Brasil
     Núcleo de Neurociências
    % =========================================================================
    """
    #signal spectrum
    ff = np.fft.fft(f)
    ff = abs(ff[0:int(np.ceil(ff.size/2))])#one-sided magnitude
    #extract boundaries of Fourier Segments
    boundaries = boundaries*np.pi/round(ff.size)
    #-- for three filters based optimziation, the first boundary should be the lower one.
    if filter_num == 3 and boundaries[0]>boundaries[1]:
        boundaries[0], boundaries[1] =  boundaries[1], boundaries[0] 
    #Filtering
    #extend the signal by mirroring to deal with boundaries
    ltemp = int(np.ceil(f.size/2)) #to behave the same as matlab's round
    fMirr =  np.append(np.flip(f[0:ltemp-1],axis = 0),f)  
    fMirr = np.append(fMirr,np.flip(f[-ltemp-1:-1],axis = 0))
    ffMirr = np.fft.fft(fMirr)
    
    #build the corresponding filter bank
    mfb = EWT_Meyer_FilterBank(boundaries,ffMirr.size)
    
    #filter the signal to extract each subband
    ewt = np.zeros(mfb.shape)
    for k in range(mfb.shape[1]):
        ewt[:,k] = np.real(np.fft.ifft(np.conjugate(mfb[:,k])*ffMirr))  
    ewt = ewt[ltemp-1:-ltemp,:]
    
    return ewt, mfb ,boundaries 

 
def EWT_Boundaries_Completion(boundaries,NT):
    """
    ======================================================================
    boundaries=EWT_Boundaries_Completion(boundaries,NT)

    This function permits to complete the boundaries vector to get a 
    total of NT boundaries by equally splitting the last band (highest
    frequencies)

    Inputs:
      -boundaries: the boundaries vector you want to complete
      -NT: the total number of boundaries wanted

    Output:
      -boundaries: the completed boundaries vector

    Author: Jerome Gilles
    Institution: UCLA - Department of Mathematics
    Year: 2013
    Version: 1.0
    
    Python Version: Vinícius Rezende Carvalho - vrcarva@ufmg.br
    Universidade Federal de Minas Gerais - Brasil
    Núcleo de Neurociências     
    %======================================================================
    """
    Nd=NT-len(boundaries)
    deltaw=(np.pi-boundaries[-1])/(Nd+1)
    for k in range(Nd):
        boundaries = np.append(boundaries,boundaries[-1]+deltaw)

def EWT_Meyer_FilterBank(boundaries,Nsig):
    """
     =========================================================================
     function mfb=EWT_Meyer_FilterBank(boundaries,Nsig)
    
     This function generate the filter bank (scaling function + wavelets)
     corresponding to the provided set of frequency segments
    
     Input parameters:
       -boundaries: vector containing the boundaries of frequency segments (0
                    and pi must NOT be in this vector)
       -Nsig: signal length
    
     Output:
       -mfb: cell containing each filter (in the Fourier domain), the scaling
             function comes first and then the successive wavelets
    
     Author: Jerome Gilles
     Institution: UCLA - Department of Mathematics
     Year: 2012
     Version: 1.0
     
     Python Version: Vinícius Rezende Carvalho - vrcarva@ufmg.br
     Universidade Federal de Minas Gerais - Brasil
     Núcleo de Neurociências 
     =========================================================================
     """
    Npic = len(boundaries)
    #compute gamma
    gamma = 1
    for k in range(Npic-1):
        r = (boundaries[k+1]-boundaries[k])/ (boundaries[k+1]+boundaries[k])
        if r < gamma:
            gamma = r
    r = (np.pi - boundaries[Npic-1])/(np.pi + boundaries[Npic-1])
    if r <gamma:
        gamma = r
    gamma = (1-1/Nsig)*gamma#this ensure that gamma is chosen as strictly less than the min

    
    mfb = np.zeros([Nsig,Npic+1])

    #EWT_Meyer_Scaling
    Mi=int(np.floor(Nsig/2))
    w=np.fft.fftshift(np.linspace(0,2*np.pi - 2*np.pi/Nsig,num = Nsig))
    w[0:Mi]=-2*np.pi+w[0:Mi]
    aw=abs(w)
    yms=np.zeros(Nsig)
    an=1./(2*gamma*boundaries[0])
    pbn=(1.+gamma)*boundaries[0]
    mbn=(1.-gamma)*boundaries[0]
    for k in range(Nsig):
       if aw[k]<=mbn:
           yms[k]=1
       elif ((aw[k]>=mbn) and (aw[k]<=pbn)):
           yms[k]=np.cos(np.pi*EWT_beta(an*(aw[k]-mbn))/2)
    yms=np.fft.ifftshift(yms) 
    mfb[:,0] = yms
    
    #generate rest of the wavelets
    for k in range(Npic-1):
        mfb[:,k+1] = EWT_Meyer_Wavelet(boundaries[k],boundaries[k+1],gamma,Nsig)

    mfb[:,Npic] = EWT_Meyer_Wavelet(boundaries[Npic-1],np.pi,gamma,Nsig)
    
    return mfb


def EWT_beta(x):
    """
    Beta = EWT_beta(x)
    function used in the construction of Meyer's wavelet
    """
    if x<0:
        bm=0
    elif x>1:
        bm=1
    else:
        bm=(x**4)*(35.-84.*x+70.*(x**2)-20.*(x**3))
    return bm

def EWT_Meyer_Wavelet(wn,wm,gamma,Nsig):
    """
    =========================================================
    ymw=EWT_Meyer_Wavelet(wn,wm,gamma,N)
    
    Generate the 1D Meyer wavelet in the Fourier
    domain associated to scale segment [wn,wm] 
    with transition ratio gamma
    
    Input parameters:
      -wn : lower boundary
      -wm : upper boundary
      -gamma : transition ratio
      -N : number of point in the vector
    
    Output:
      -ymw: Fourier transform of the wavelet on the band [wn,wm]

    Author: Jerome Gilles
    Institution: UCLA - Department of Mathematics
    Year: 2012
    Version: 1.0
    
    Python Version: Vinícius Rezende Carvalho - vrcarva@ufmg.br
    Universidade Federal de Minas Gerais - Brasil
    Núcleo de Neurociências 
    ==========================================================            
    """
    Mi=int(np.floor(Nsig/2))
    w=np.fft.fftshift(np.linspace(0,2*np.pi - 2*np.pi/Nsig,num = Nsig))
    w[0:Mi]=-2*np.pi+w[0:Mi]
    aw=abs(w)
    ymw=np.zeros(Nsig)
    an=1./(2*gamma*wn)
    am=1./(2*gamma*wm)
    pbn=(1.+gamma)*wn
    mbn=(1.-gamma)*wn
    pbm=(1.+gamma)*wm
    mbm=(1.-gamma)*wm

    for k in range(Nsig):
       if ((aw[k]>=pbn) and (aw[k]<=mbm)):
           ymw[k]=1
       elif ((aw[k]>=mbm) and (aw[k]<=pbm)):
           ymw[k]=np.cos(np.pi*EWT_beta(am*(aw[k]-mbm))/2)
       elif ((aw[k]>=mbn) and (aw[k]<=pbn)):
           ymw[k]=np.sin(np.pi*EWT_beta(an*(aw[k]-mbn))/2)

    ymw=np.fft.ifftshift(ymw)
    return ymw
            








    



