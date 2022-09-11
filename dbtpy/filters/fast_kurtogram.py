
#------------------------------------------------------------------------------
# The old python version, link: https://github.com/amaggi/seismokurt/blob/master/originals/Fast_Kurtogram.py
# This python version is relatively old (2012), which may raise many errors for now (2021)
# In addtion, the old python version also has some differences with the latest matlab version
# A new version is created below by mozhenling (zhenlmo2-c@my.cityu.edu.hk)
# based on the old python version(2012) and the latest matlab version of fast kurtogram(2014)
# I have validated it by using the 'x.mat' data in the fast kurtogram (matlab) files
# and the results are almost the same.
#------------------------------------------------------------------------------

import numpy as np
import scipy.signal as si
import logging
import matplotlib.pyplot as plt
# import sys, os
# from obspy.signal.filter import bandpass, envelope 
from matplotlib.colors import LinearSegmentedColormap
# from seaborn import heatmap
#------------------------------------------------------------------------------
# use the following colormap to have the similar style as the matlab
#----------------------------- colormap----------------------------------------
cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

#------------------------------------------------------------------------------

eps = np.finfo(float).eps


def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i
    
def Fast_Kurtogram(x, nlevel, Fs = None):
    # % Fast_Kurtogram(x,nlevel,Fs)
    # % Computes the fast kurtogram of signal x up to level 'nlevel' via a fast decimated filterbank tree.
    # % Maximum number of decomposition levels is log2(length(x)), but it is 
    # % recommended to stay by a factor 1/8 below this.
    # % Fs = sampling frequency of signal x (default is Fs = 1)
    # %
    # % --------------------------
    # % Reference: J. Antoni, Fast Computation of the Kurtogram for the Detection of Transient Faults, 
    # % Mechanical Systems and Signal Processing, Volume 21, Issue 1, 2007, pp.108-124.
    # % --------------------------
    # % Author: J. Antoni
    # % Last Revision: 12-2014
    # % An old python version: T. Lecocq , 02-2012
    # % Latest python version: mozhenling, 06-2021
    # % --------------------------

    N = len(x)
    N2 = np.log2(N) - 7
    if nlevel > N2:
       logging.error('Please enter a smaller number of decomposition levels')

    Fs = Fs if Fs is not None else 1
    # remove the DC component
    x = x - np.mean(x)
    #--------------------------------------------------------------------------
    #-------------------- Analytic generating filters
    N, fc = 16, .4	 # a short filter is just good enough!		
    # python starts indexing at 0 while matlabe starts at 1				
    h = si.firwin(N+1,fc) * np.exp(2*1j*np.pi*np.arange(N+1)*0.125)
    n = np.arange(2,N+2)
    g = h[ (1-n)%N ]* (-1.0) **(1-n)
    N = int(np.fix((3./2.*N)))
    h1 = si.firwin(N+1,2./3*fc)*np.exp(2j*np.pi*np.arange(N+1)*0.25/3.)
    h2 = h1*np.exp(2j*np.pi*np.arange(N+1)/6.)
    h3 = h1*np.exp(2j*np.pi*np.arange(N+1)/3.)  
    # kurtosis of the complex envelope
    Kwav = K_wpQ(x,h,g,h1,h2,h3,nlevel,'kurt2')				
    # keep positive values only!
    Kwav[Kwav < 0] = 0 
    #--------------------------------------------------------------------------
    #--------------------- GRAPHICAL DISPLAY OF RESULTS
    fig = plt.figure()
    #~ plt.subplot(ratio='auto')
    Level_w = np.arange(1,nlevel+1)
    Level_w = np.array([Level_w, Level_w + np.log2(3.)-1])
    Level_w = sorted(Level_w.ravel())
    Level_w = np.append(0,Level_w[0:2*nlevel-1])
    freq_w = Fs*(np.arange(0,3*2.0**nlevel-1+1))/(3*2**(nlevel+1)) + 1.0/(3.*2.**(2+nlevel))
    #-------------------- the imagesc in python
    # parula_map = LinearSegmentedColormap.from_list('parula', Kwav)
    plt.imshow(Kwav, aspect='auto',extent=(freq_w[0],freq_w[-1],Level_w[0],Level_w[-1]),
               interpolation='none', origin = 'upper', cmap = parula_map)
    #-- process the y ticks so that it just looks like the matlab version
    ytick_pos = np.arange(0.25, 0.5*len(Level_w), 0.5)         
    ytick_lab = np.round(np.flip(Level_w * 10))/10
    plt.yticks(ticks = ytick_pos , labels = ytick_lab)
    plt.ylim([Level_w[0],Level_w[-1]])
    plt.xlim([freq_w[0],freq_w[-1]])
    # raise ValueError
    plt.xlabel('frequency [Hz]')
    plt.ylabel('level k')
    
    # the maximum value and the indexes of row and col.
    [I,J,M] = max_IJ(Kwav)

    fi = J /3./2**(nlevel+1) # start indexing at 0
    fi += 2.**(-2-Level_w[I])
    #---- keep decimal : you may use 'format' instead 
    # round and keep one decimal of maximum kurtosis
    M_kurt_max = round(M, 1)
    level_kurt_max = np.fix(10*Level_w[I])/10
    fc_kurt_max  = round(Fs*fi, 4)
    Bw = round( Fs * 2 **(- Level_w[I] -1 ), 4)
    
    print('----- Please use and type the info. shown below -----')
    print('kurt_max = '+ str( M_kurt_max ) + ' @level ' 
          + str(level_kurt_max) +  ' Bw = '+ str(Bw) + ' Hz'+' fc = ' + str( fc_kurt_max ) + ' Hz')
    
    plt.title('kurt_max = '+ str( M_kurt_max ) +
              ' @level ' + str(level_kurt_max) +  
              ' Bw = '+ str(Bw) + ' Hz'+ ' fc = ' + str( fc_kurt_max ) + ' Hz' )
    
    plt.colorbar()
    plt.show()
    
    #--------------------------------------------------------------------------
    #--- interaction 
    #Ajouter le signal filtering !
    c = [];
    test = int(input('Do you want to filter out transient signals from the kurtogram (yes = 1 ; no = 0): '))


    while test == 1:
        fi = float(input('	Enter the optimal carrier frequency where to filter the signal: '))
        fi = fi/Fs
        # fi = fi
       
        lev = int(input('	Enter the optimal level where to filter the signal: '))
        lev = lev
    
        c,Bw,fc = Find_wav_kurt(x,h,g,h1,h2,h3,nlevel,lev,fi,'kurt2',Fs)
    #~ else
            #~ [c,Bw,fc] = Find_wav_kurt(x,h,g,h1,h2,h3,nlevel,lev,fi,'kurt1',Fs);
        test = int(input('Do you want to keep on filtering out transients (yes = 1 ; no = 0): '))
    #--------------------------------------------------------------------------
    
    lev = level_kurt_max

    c,Bw,fc = Find_wav_kurt(x,h,g,h1,h2,h3,nlevel,lev,fi,'kurt2',Fs)
    
    return c

def max_IJ(Kwav):
    # get the maximum value 
    M = np.max(Kwav)
    # get the index of row and col.
    index = np.argmax(Kwav)
    index = np.unravel_index(index, Kwav.shape)
    I = index[0] # row
    J = index[1] # col.
    return [I,J,M]    

def K_wpQ(x,h,g,h1,h2,h3,nlevel,opt,level=0):
    # K = K_wpQ(x,h,g,h1,h2,h3,nlevel)
    # Calculates the kurtosis K of the complete quinte wavelet packet transform w of signal x, 
    # up to nlevel, using the lowpass and highpass filters h and g, respectively. 
    # The WP coefficients are sorted according to the frequency decomposition.
    # This version handles both real and analytical filters, but does not yiels WP coefficients
    # suitable for signal synthesis.
    #
    # -----------------------
    # J Antoni : 12/2004 
    # Translation: T. Lecocq 02/2012
    # Latest python version: mozhenling,06-2021
    # -----------------------   
    L = np.floor(np.log2(len(x)))
    if level == 0:
        if nlevel >= L:
            logging.error('nlevel must be smaller')
        level=nlevel
    x = x.ravel()
    #~ print "THIS"
    #~ print h, g
    KD, KQ = K_wpQ_local(x,h,g,h1,h2,h3,nlevel,opt,level)
    K = np.zeros((2*nlevel,3*2**nlevel))
    #~ print "******************************************************"
    #~ print KD.shape, KQ.shape, K.shape
    KD = KD
    KQ = KQ

    K[0,:] = KD[0,:]
    for i in range(nlevel-1):
        #~ print K[2*i,:].shape
        K[2*i + 1,:] = KD[i+1,:] 
        #~ print K[2*i+1,:].shape
        K[2*i + 2,:] = KQ[i,:]
    
   

    K[2*nlevel-1,:] = KD[nlevel,:]
    #~ print "K Final Shape", K.shape
    # raise ValueError
    return K

def K_wpQ_local(x,h,g,h1,h2,h3,nlevel,opt,level):
    # print ("LEVEL", level)
    a,d = DBFB(x,h,g)
    
    N = len(a)
    d = d*np.power(-1.,np.arange(1,N+1))
    K1 = kurt(a[len(h)-1:],opt)
    K2 = kurt(d[len(g)-1:],opt)
    if level > 2:
        a1,a2,a3 = TBFB(a,h1,h2,h3)
        d1,d2,d3 = TBFB(d,h1,h2,h3)
        Ka1 = kurt(a1[len(h)-1:],opt)
        Ka2 = kurt(a2[len(h)-1:],opt)
        Ka3 = kurt(a3[len(h)-1:],opt)
        Kd1 = kurt(d1[len(h)-1:],opt)
        Kd2 = kurt(d2[len(h)-1:],opt)
        Kd3 = kurt(d3[len(h)-1:],opt)
    else:
        Ka1 = 0
        Ka2 = 0
        Ka3 = 0
        Kd1 = 0
        Kd2 = 0
        Kd3 = 0
    
    if level ==1:
        #~ print "level = 1"
        K =np.array([K1*np.ones(3),K2*np.ones(3)]).flatten()
        #~ print 'K.shape',K.shape
        KQ = np.array([Ka1,Ka2,Ka3,Kd1,Kd2,Kd3])
        #~ print 'KQ.shape',KQ.shape
    if level > 1:
        #~ print "entering rec with level %i"%(level-1)
        #~ print "doing A"
        Ka,KaQ = K_wpQ_local(a,h,g,h1,h2,h3,nlevel,opt,level-1)
        #~ print "doing D"
        Kd,KdQ = K_wpQ_local(d,h,g,h1,h2,h3,nlevel,opt,level-1)
        #~ print "out of rec level %i" % (level -1)
        #~ print Ka.shape, Kd.shape
        K1 = K1*np.ones(np.max(Ka.shape))
        K2 = K2*np.ones(np.max(Kd.shape))
        K12 = np.append(K1,K2)
        Kad = np.hstack((Ka, Kd))
        #~ print ">", K12.shape, Kad.shape
        K = np.vstack((K12,Kad))
        
        Long = int(2./6*np.max(KaQ.shape))
        
        Ka1 = Ka1*np.ones(Long)
        Ka2 = Ka2*np.ones(Long)
        Ka3 = Ka3*np.ones(Long)
        Kd1 = Kd1*np.ones(Long)
        Kd2 = Kd2*np.ones(Long)
        Kd3 = Kd3*np.ones(Long)
        tmp = np.hstack((KaQ,KdQ))
        #~ print "HEEEERE"
        #~ print tmp.shape
        KQ = np.concatenate((Ka1,Ka2,Ka3,Kd1,Kd2,Kd3))
        KQ = np.vstack((KQ, tmp))
        #~ if tmp.shape[0] != KQ.shape[0]:
            #~ tmp = tmp.T
        #~ for i in range(tmp.shape[0]):
            #~ KQ = np.vstack((KQ,tmp[i]))
        
        #~ print "4", K.shape, KQ.shape
        

    
    if level == nlevel:
        K1 = kurt(x,opt)
        K = np.vstack((K1*np.ones(np.max(K.shape)), K))
        #~ print "K shape", K.shape

        a1,a2,a3 = TBFB(x,h1,h2,h3)
        Ka1 = kurt(a1[len(h)-1:],opt)
        Ka2 = kurt(a2[len(h)-1:],opt)
        Ka3 = kurt(a3[len(h)-1:],opt)
        
        Long = int(1./3*np.max(KQ.shape))

        Ka1 = Ka1*np.ones(Long)
        Ka2 = Ka2*np.ones(Long)
        Ka3 = Ka3*np.ones(Long)
        # print (KQ.shape)
        tmp = np.array(KQ[0:-2])
        #~ print "level==nlevel"
        
        KQ = np.concatenate((Ka1,Ka2,Ka3))
        KQ = np.vstack((KQ,tmp))
    
    #~ print "i'm leaving level=%i and K.shape="%level,K.shape, "and KQ.shape=",KQ.shape
    return K, KQ

def kurt(x, opt):
    if opt=='kurt2':
        if np.all(x==0):
            K=0
            E=0
            return K
        x -= np.mean(x)
        E = np.mean(np.abs(x)**2)
        if E < eps:
            K=0
            return K
        K = np.mean(np.abs(x)**4)/E**2
        if np.all(np.isreal(x)):
            K = K - 3
        else:
            K = K - 2
    return K

def DBFB(x,h,g):
    # Double-band filter-bank.
    #   [a,d] = DBFB(x,h,g) computes the approximation
    #   coefficients vector a and detail coefficients vector d,
    #   obtained by passing signal x though a two-band analysis filter-bank.
    #   h is the decomposition low-pass filter and
    #   g is the decomposition high-pass filter.
    
    N = len(x)
    La = len(h)
    Ld = len(g)

    # lowpass filter
    a = si.lfilter(h,1,x)
    a = a[1::2]
    a = a.ravel()

    # highpass filter
    d = si.lfilter(g,1,x)
    d = d[1::2]
    d = d.ravel()
    # raise ValueError
    return (a,d)

def TBFB(x,h1,h2,h3):
    # Trible-band filter-bank.
    #   [a1,a2,a3] = TBFB(x,h1,h2,h3) 
    
    N = len(x)
    La1 = len(h1)
    La2 = len(h2)
    La3 = len(h3)

    # lowpass filter
    a1 = si.lfilter(h1,1,x)
    a1 = a1[2::3]
    a1 = a1.ravel()

    # passband filter
    a2 = si.lfilter(h2,1,x)
    a2 = a2[2::3]
    a2 = a2.ravel()

    # highpass filter
    a3 = si.lfilter(h3,1,x)
    a3 = a3[2::3]
    a3 = a3.ravel()
    return (a1,a2,a3)

def Find_wav_kurt(x,h,g,h1,h2,h3,nlevel,Sc,Fr,opt,Fs=1):
    # [c,Bw,fc,i] = Find_wav_kurt(x,h,g,h1,h2,h3,nlevel,Sc,Fr,opt2)
    # Sc = -log2(Bw)-1 with Bw the bandwidth of the filter
    # Fr is in [0 .5]
    #
    # -------------------
    # J. Antoni : 12/2004
    # -------------------
    level = np.fix((Sc))+ ((Sc%1) >= 0.5) * (np.log2(3)-1)
    Bw = 2**(-level-1)
    freq_w = np.arange(0,2**(level))/ 2**(level+1) + Bw/2. #   np.arange(0,2**(level-1))
    J = np.argmin(np.abs(freq_w-Fr))
    fc = freq_w[J]
    i = int(np.round(fc/Bw-1./2))
    if level % 1 == 0:
        acoeff = binary(i, level)
        bcoeff = np.array([])
        temp_level = level
    else:
        i2 = np.fix((i/3.))
        temp_level = np.fix((level))-1
        acoeff = binary(i2,temp_level)
        bcoeff = np.array([i-i2*3,])
    acoeff = acoeff[::-1]
    # raise ValueError
    c = K_wpQ_filt(x,h,g,h1,h2,h3,acoeff,bcoeff,temp_level)
    kx = kurt(c,opt)
    # raise ValueError
    # print ("kx", kx)
    
    sig = np.median(np.abs(c))/np.sqrt(np.pi/2.)
    # print (sig)
    threshold = sig*raylinv(np.array([.999,]),np.array([1,]))
    # print ("threshold", threshold)
    #~ spec = int(raw_input('	Do you want to see the envelope spectrum (yes = 1 ; no = 0): '))
    spec = 1
    
    t = np.arange(len(x))/Fs
    tc = np.linspace(t[0],t[-1],len(c))
    
    fig = plt.figure()
    ax1 = plt.subplot(2+spec,1,1)
    plt.plot(t,x,'k',label='Original Signal')
    #~ plt.plot(tc,(c/np.max(c))*np.max(np.abs(x)),c='r',label='Filtered Signal')

    # plt.plot(t,bandpass(x,Fr,Fr+50./32,Fs,corners=2),label='Obspy Filterded Signal')
    
    plt.legend()
    plt.grid(True)
    plt.subplot(2+spec,1,2,sharex=ax1)
    plt.plot(tc,np.abs(c),'k')
    #~ plt.plot(tc,envelope(c),'k')
    #~ plt.plot(tc,threshold*np.ones(len(c)),'--r')
    plt.axhline(threshold, c='r')
    
    
    # for ti in tc[ np.where(np.abs(c) >= threshold)[0]]:
    #     plt.axvline(ti,c='g',zorder=-1)
    #     ax1.axvline(ti,c='g',zorder=-1)
    
    #~ plt.title('Envlp of the filtr sgl, Bw=Fs/2^{'+(level+1)+'}, fc='+(Fs*fc)+'Hz, Kurt='+(np.round(np.abs(10*kx))/10)+', \alpha=.1%']
    plt.xlabel('time [s]')
    plt.grid(True)
    if spec == 1:
        #~ print nextpow2(len(c))
        nfft = int(nextpow2(len(c)))
        env = np.abs(c)**2
        env = env.ravel()
        env = env - np.mean(env) # remove the DC component
        S = np.abs(np.fft.fft(env*np.hanning(len(env))/len(env),nfft))
        
        #---------------------------------------------------------------------
        #--// is added
        f = np.linspace(0, 0.5*Fs/2**level, nfft//2)
        #---------------------------------------------------------------------
        plt.subplot(313)
        #---------------------------------------------------------------------
        #--// is added
        plt.plot(f,S[:nfft//2],'k')
        #---------------------------------------------------------------------
        plt.title('Fourier transform magnitude of the squared envelope')
        plt.xlabel('frequency [Hz]')
        plt.grid(True)
    plt.show()
    return [c,Bw,fc]

def binary(i,k):
    # return the coefficients of the binary expansion of i:
    # i = a(1)*2^(k-1) + a(2)*2^(k-2) + ... + a(k)

    if i>=2**k:
        logging.error('i must be such that i < 2^k !!')

    k = int(k)
    a = np.zeros(k)

    #~ print a.shape
    temp = i
    for l in np.arange(k-1,-1,-1): # bug is at here
        a[k-l-1] = np.fix(temp/2**l)
        temp = temp - a[k-l-1]*2**l

    return a
    
def K_wpQ_filt(x,h,g,h1,h2,h3,acoeff,bcoeff,level=0):
    # c = K_wpQ_filt(x,h,g,h1,h2,h3,acoeff,bcoeff,level)
    # Calculates the kurtosis K of the complete quinte wavelet packet transform w of signal x, 
    # up to nlevel, using the lowpass and highpass filters h and g, respectively. 
    # The WP coefficients are sorted according to the frequency decomposition.
    # This version handles both real and analytical filters, but does not yiels WP coefficients
    # suitable for signal synthesis.
    #
    # -----------------------
    # J Antoni : 12/2004 
    # -----------------------   
    nlevel = len(acoeff)
    L = np.floor(np.log2(len(x)))
    if level==0:
        if nlevel >= L:
            logging.error('nlevel must be smaller !!')
        level = nlevel
    x = x.ravel()
    if nlevel == 0:
        if np.empty(bcoeff):
            c = x
        else:
            c1, c2, c3 = TBFB(x,h1,h2,h3)
            if bcoeff == 0:
                c = c1[len(h1)-1:]
            elif bcoeff == 1:
                c = c2[len(h2)-1:]
            elif bcoeff == 2:
                c = c3[len(h3)-1:]
    else:
        c = K_wpQ_filt_local(x,h,g,h1,h2,h3,acoeff,bcoeff,level)
    return c

def  K_wpQ_filt_local(x,h,g,h1,h2,h3,acoeff,bcoeff,level):
    # print (level, x[:10])
    # debug: DBTB is correct
    # acoeff is not correct
    a, d = DBFB(x,h,g)         # perform one analysis level into the analysis tree
   
    N = len(a)                       
    d = d*np.power(-1.,np.arange(1,N+1))
    level = int(level)
    if level == 1:
        #~ print "bcoeff", bcoeff
        if len(bcoeff) ==0:
          if acoeff[level-1] == 0:
             c = a[len(h)-1:]
          else:
             c = d[len(g)-1:]
        else:
            if acoeff[level-1] == 0:
                c1,c2,c3 = TBFB(a,h1,h2,h3)
            else:
                c1,c2,c3 = TBFB(d,h1,h2,h3)
            if bcoeff == 0:
                c = c1[len(h1)-1:]
            elif bcoeff == 1:
                c = c2[len(h2)-1:]
            elif bcoeff == 2:
                c = c3[len(h3)-1:]
    if level > 1:
        #~ print "acoeff", acoeff[level-1]
        if acoeff[level-1] == 0:
            c = K_wpQ_filt_local(a,h,g,h1,h2,h3,acoeff,bcoeff,level-1)
        else:
            c = K_wpQ_filt_local(d,h,g,h1,h2,h3,acoeff,bcoeff,level-1)
    # print ('kurt', kurt(c,'kurt2'))
    #~ print 'c.shape', c.shape
    return c

def raylinv(p,b):
    #RAYLINV  Inverse of the Rayleigh cumulative distribution function (cdf).
    #   X = RAYLINV(P,B) returns the Rayleigh cumulative distribution 
    #   function with parameter B at the probabilities in P.

    #~ if nargin <  1: 
        #~ logging.error('Requires at least one input argument.') 

    # Initialize x to zero.
    x = np.zeros(len(p))
    # Return NaN if the arguments are outside their respective limits.
    k = np.where(((b <= 0)| (p < 0)| (p > 1)))[0]
    
    if len(k) != 0: 
        tmp  = np.NaN
        x[k] = tmp(len(k))

    # Put in the correct values when P is 1.
    k = np.where(p == 1)[0]
    #~ print k
    if len(k)!=0:
        tmp  = np.Inf
        x[k] = tmp(len(k))

    k = np.where(((b > 0) & (p > 0) & (p < 1)))[0]
    #~ print k
    
    if len(k)!=0:
        pk = p[k]
        bk = b[k]
        #~ print pk, bk
        x[k] = np.sqrt((-2*bk ** 2) * np.log(1 - pk))
    return x


if __name__ == "__main__":
    from scipy.io.matlab import loadmat
    from librosa import lpc
    from scipy.signal import convolve
    # from dbtpy.tools.time_freq_tools import tf_one_show,tf_two_show
    # from scipy.signal import filtfilt # (version 1.6.3)
    #~ v1 = loadmat(r"C:\Users\tlecocq\Documents\Tom's Share\Pack Kurtogram\Pack Kurtogram V3\VOIE1.mat")
    #~ x = v1['v1']
    #-------------------- su data
    # from dbtpy.data.suzhouU.su_data import SuData
    # data = SuData()
    # sig_kargs={}
    # sig_kargs['fault_str'] = 'Roller'
    # sig_kargs['resolution'] = 0.5
    # sig_kargs['path'] = r'C:\Users\MSI-NB\Desktop\dbtree\dbtpy\data\suzhouU\su0.2mm1kN-InnerRollingOuter-Fs10k-fIndex2-Ns102400.mat'
    # sig = data.load(**sig_kargs)
    # x = sig
    # Fs = 10e3
    # nlevel= 7

    #-- kurtogram data
    x_dict = loadmat('x.mat')
    x = x_dict['x'].ravel()
    Fs = 1
    nlevel = 7
    
    # #--------------------------------------------------------------------------
    # #-------------------- pre-whitening
    # #-- Pre-whitening of the signal by AR filtering (optional)
    # #   (always helpful in detection problems)
    # #-- you need pip install scipy
    # x = x - np.mean(x)
    # Na = 100 # Order of the linear filter
    # #-- Linear Prediction Coefficients via Burg’s method
    # a = lpc(x,Na) 
    # x = convolve(a,x) # equivalence to fftfilt in matlab
    # x = x[Na:-100]
    # # validation
    # # tf_two_show(1, x)
    # #--------------------------------------------------------------------------
    
    c = Fast_Kurtogram(x, nlevel, Fs)