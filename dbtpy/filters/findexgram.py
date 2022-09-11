"""
Created on Thu Jun  12 15:50:28 2021

@author: mozhenling
"""
import pylab as pl
import numpy as np
import scipy.signal as si
import logging
import matplotlib.pyplot as plt
# import sys, os
# from obspy.signal.filter import bandpass, envelope 
from matplotlib.colors import LinearSegmentedColormap
from dbtpy.tools.colorMap import cm_data

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
#------------------------------------------------------------------------------
eps = np.finfo(float).eps
#------------------------------------------------------------------------------
#-- redefine a non negative fault index here 
def findex(x, findex_dict, level=0):
    fdict = findex_dict
    if fdict['findex_kwargs']['findexBase'] == 'kurt' \
        and (fdict['findex_kwargs']['sigD'] is None or fdict['findex_kwargs']['sigD'] =='env'):
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
    elif fdict['findex_kwargs']['findexBase'] in [ 'vSNR', 'harL2L1', 'CHNR']:
        nfft = int(nextpow2(len(x))) 
        fre2seq = nfft * 2 **level / fdict['findex_kwargs']['fs']
        K= fdict['findex_fun'](x, fre2seq = fre2seq, **fdict['findex_kwargs']) 
    else:
        # try:
        K= fdict['findex_fun'](x, **fdict['findex_kwargs']) 
        # except:
        #     raise ValueError

    return K
#------------------------------------------------------------------------------
#-- for downsampling
def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i
    
def findexgram(sig, nlevel, findex_dict=None, fs = None, heatmap_show = False, other_show = False):
    """
    the fault/fast index grams when replacing kurtosis with other fault indexes
    
    inputs:
        -sig             # the real valued signal 
        -nlevel          # the decomposition level
        -fs              # sampling freqeuncy
    
    output:
        -c               # the filtered band signal with the largest value of fault index 
    
    """
    # % Fast_findexogram(x,nlevel,Fs)
    # % Computes the fast findexogram of signal x up to level 'nlevel' via a fast decimated filterbank tree.
    # % Maximum number of decomposition levels is log2(length(x)), but it is 
    # % recommended to stay by a factor 1/8 below this.
    # % Fs = sampling frequency of signal x (default is Fs = 1)
    # %
    # % --------------------------
    # % Reference: J. Antoni, Fast Computation of the findexogram for the Detection of Transient Faults, 
    # % Mechanical Systems and Signal Processing, Volume 21, Issue 1, 2007, pp.108-124.
    # % --------------------------
    # % Author: J. Antoni
    # % Last Revision: 12-2014
    # % An old python version: T. Lecocq , 02-2012
    # % Latest python version: mozhenling, 06-2021
    # % --------------------------
    
    #-- keep conventions
    x = sig
    Fs = fs
    
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
    # findexosis of the complex envelope
      
    Kwav = K_wpQ(x,h,g,h1,h2,h3,nlevel,findex_dict)				
    # keep positive values only!
    Kwav[Kwav < 0] = 0 
    
    #--------------------------------------------------------------------------
    #--------------------- GRAPHICAL DISPLAY OF RESULTS
    
    #~ plt.subplot(ratio='auto')
    Level_w = np.arange(1,nlevel+1)
    Level_w = np.array([Level_w, Level_w + np.log2(3.)-1])
    Level_w = sorted(Level_w.ravel())
    Level_w = np.append(0,Level_w[0:2*nlevel-1])
    freq_w = Fs*(np.arange(0,3*2.0**nlevel-1+1))/(3*2**(nlevel+1)) + 1.0/(3.*2.**(2+nlevel))
    #-------------------- the imagesc in python
    #-- show heatmap
    if heatmap_show:
        fig = plt.figure()
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
    # round and keep one decimal of maximum findexosis
    M_findex_max = round(M, 1)
    level_findex_max = round(10*Level_w[I])/10
    fc_findex_max  = round(Fs*fi, 2)
    Bw = round( Fs * 2 **(- Level_w[I] - 1), 2)
    
    # print('----- Please use and type the info. shown below -----')
    # print('findex_max = '+ str( M_findex_max ) + ' @level ' 
    #       + str(level_findex_max) +  ' Bw = '+ str(Bw) + ' Hz'+' fc = ' + str( fc_findex_max ) + ' Hz')
    if findex_dict['findex_kwargs']['sigD']  is None:
        findex_dict['findex_kwargs']['sigD'] = ''
    if heatmap_show:
        plt.title(findex_dict['findex_kwargs']['findexBase'] + '_'
                  + findex_dict['findex_kwargs']['sigD'] + '_max='
                  + str( M_findex_max ) +
                  ' @level ' + str(level_findex_max) +  
                  ' Bw = '+ str(Bw) + ' Hz'+ ' fc = ' + str( fc_findex_max ) + ' Hz' )
        plt.colorbar()
        plt.show()
    
    lev = level_findex_max

    # you may also return the Bw and fc 
    ([c, _, _], [ses_x, ses_y] ) = Find_wav_findex(x,h,g,h1,h2,h3,nlevel,lev,fi,findex_dict,Fs, other_show)

    return ([M, level_findex_max, Bw, fc_findex_max], [c, ses_x, ses_y], [Kwav, freq_w, Level_w] ) 

def show_heatmap(Kwav, freq_w, Level_w, nlevel, fs, findex_str=None, figsize = (4, 2), dpi = 144,
             fig_save_path= None, fig_format = 'png', fontsize = 7 ):
    """
    customize the heatmap
    """
    #-- convert list object to array object 
    Kwav = np.array(Kwav)
    freq_w = np.array(freq_w)
    Level_w = np.array(Level_w)
  
    findex_str = findex_str + '_max' if findex_str is not None else 'findex_max'
    
    #-- figure size 
    plt.figure(figsize=figsize, dpi=dpi)
    #-- font size
    plt.rc('font', size = fontsize)
    #-- plot heat map    
    plt.imshow(Kwav, aspect='auto',extent=(freq_w[0],freq_w[-1],Level_w[0],Level_w[-1]),
               interpolation='none', origin = 'upper', cmap = parula_map)
    
    #-- process the y ticks so that it just looks like the matlab version
    # ytick_pos = np.arange(Level_w[0]+0.25, Level_w[-1]+0.75, 0.5)
    ytick_pos = np.arange(0, 0.5*(len(Level_w)), 0.5) + 0.25
    ytick_lab = np.round(np.flip(Level_w * 10))/10
    plt.yticks(ticks = ytick_pos , labels = ytick_lab)
    plt.ylim([Level_w[0],Level_w[-1]])
    plt.xlim([freq_w[0],freq_w[-1]])
    
    # raise ValueError
    plt.xlabel('Frequency (H)', fontsize = fontsize + 1)
    plt.ylabel('Level k', fontsize = fontsize + 1)
    
     # the maximum value and the indexes of row and col.
    [I,J,M] = max_IJ(Kwav)

    fi = J /3./2**(nlevel+1) # start indexing at 0
    fi += 2.**(-2-Level_w[I])
    #---- keep decimal : you may use 'format' instead 
    # round and keep one decimal of maximum findexosis
    M_findex_max = round(M, 1)
    level_findex_max = round(10*Level_w[I])/10
    fc_findex_max  = round(fs*fi, 1)
    Bw = round( fs * 2 **(- Level_w[I] - 1), 1)
    
    plt.title(findex_str + ': '+ str( M_findex_max ) +
              ' @level: ' + str(level_findex_max) +  
              ' Bw: '+ str(Bw) + ' Hz'+ ' fc: ' + str( fc_findex_max ) + ' Hz', fontsize = fontsize + 1 )
   
    plt.colorbar()
    plt.tight_layout()
    if fig_save_path is not None:
        plt.savefig(fig_save_path, format=fig_format)
    plt.show() 

def show_sses_xy(ses_x, ses_y, SSES=True, title='', xlabel = 'Frequency (Hz)',
             ylabel = 'Normalized amplitude', f_target=None, figsize = (3.5, 1.8), dpi = 144,
             fig_save_path= None, fig_format = 'png', fontsize = 8, linewidth = 1 ):
    """
    Show the square of the squared envelope spectrum given ses_x, and ses_y if sses is true
    Otherwise, show ses
    
    """
    #-- normalized by the maximum amplitude
    sesMax = max(ses_y) 
    ses_y = ses_y /sesMax

    (sses_y, label) = (ses_y**2, 'SES' ) if SSES else (ses_y, 'SES') # you may use sses instead
    sses_x = ses_x

    plt.figure(figsize=figsize, dpi=dpi)
    plt.rc('font', size = fontsize)
    
    if f_target is not None:
        harN = 5
        harColor = ['r', 'r', 'r','m','m']
        harLine = ['--', '-.', ':','--','-.']
        point_num = 10
        targetHarAmp = [np.linspace(0, 1.1, point_num ) for i in range(harN) ]
        targetHar = [[f_target + i*f_target for j in range(point_num)] for i in range(harN) ]
        
        for i, (tar, tarAmp) in enumerate(zip(targetHar, targetHarAmp)):
            plt.plot(tar, tarAmp, harColor[i] + harLine[i],  label ='Har'+str(i+1), linewidth=linewidth + 0.2 )
            # raise ValueError
            plt.ylim([0, 1.1])
            plt.xlim([0, 7 * f_target])
            
        plt.plot(sses_x, sses_y, 'b', label = label, linewidth=linewidth)
        #-- font size of the figure if not given otherwise
        # plt.xticks(fontsize = fontsize)
        # plt.yticks(fontsize = fontsize)
        ax = pl.gca()
        ax.yaxis.get_major_formatter().set_powerlimits((0,1))
        plt.xlabel(xlabel, fontsize = fontsize + 0.5)
        plt.ylabel(ylabel, fontsize = fontsize + 0.5)
        plt.title(title, fontsize = fontsize + 1)
        plt.legend(fontsize = fontsize - 1)
        
        plt.tight_layout()
        if fig_save_path is not None:
            plt.savefig(fig_save_path, format=fig_format)
        plt.show()       

def max_IJ(Kwav):
    # get the maximum value 
    M = np.max(Kwav)
    # get the index of row and col.
    index = np.argmax(Kwav)
    index = np.unravel_index(index, Kwav.shape)
    I = index[0] # row
    J = index[1] # col.
    return [I,J,M]    

def K_wpQ(x,h,g,h1,h2,h3,nlevel,findex_dict,level=0):
    # K = K_wpQ(x,h,g,h1,h2,h3,nlevel)
    # Calculates the findexosis K of the complete quinte wavelet packet transform w of signal x, 
    # up to nlevel, using the lowpass and highpass filters h and g, respectively. 
    # The WP coefficients are sorted according to the frequency decomposition.
    # This version handles both real and analytical filters, but does not yiels WP coefficients
    # suitable for signal synthesis.
    #
    # -----------------------
    # J Antoni : 12/2004 
    # Translation: T. Lecocq 02/2012
    # Latest python version: mozhenling,06/2021
    # -----------------------   
    L = np.floor(np.log2(len(x)))
    if level == 0:
        if nlevel >= L:
            logging.error('nlevel must be smaller')
        level=nlevel
    x = x.ravel()
    #~ print "THIS"
    #~ print h, g
    KD, KQ = K_wpQ_local(x,h,g,h1,h2,h3,nlevel,findex_dict,level)
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

def K_wpQ_local(x,h,g,h1,h2,h3,nlevel,findex_dict,level):
    # print ("LEVEL", level)
    a,d = DBFB(x,h,g)
    
    N = len(a)
    d = d*np.power(-1.,np.arange(1,N+1))
    K1 = findex(a[len(h)-1:],findex_dict, level)
    K2 = findex(d[len(g)-1:],findex_dict, level)
    if level > 2:
        a1,a2,a3 = TBFB(a,h1,h2,h3)
        d1,d2,d3 = TBFB(d,h1,h2,h3)
        Ka1 = findex(a1[len(h)-1:],findex_dict, level)
        Ka2 = findex(a2[len(h)-1:],findex_dict, level)
        Ka3 = findex(a3[len(h)-1:],findex_dict, level)
        Kd1 = findex(d1[len(h)-1:],findex_dict, level)
        Kd2 = findex(d2[len(h)-1:],findex_dict, level)
        Kd3 = findex(d3[len(h)-1:],findex_dict, level)
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
        Ka,KaQ = K_wpQ_local(a,h,g,h1,h2,h3,nlevel,findex_dict,level-1)
        #~ print "doing D"
        Kd,KdQ = K_wpQ_local(d,h,g,h1,h2,h3,nlevel,findex_dict,level-1)
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
        K1 = findex(x,findex_dict, level)
        K = np.vstack((K1*np.ones(np.max(K.shape)), K))
        #~ print "K shape", K.shape

        a1,a2,a3 = TBFB(x,h1,h2,h3)
        Ka1 = findex(a1[len(h)-1:],findex_dict, level)
        Ka2 = findex(a2[len(h)-1:],findex_dict, level)
        Ka3 = findex(a3[len(h)-1:],findex_dict, level)
        
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

def Find_wav_findex(x,h,g,h1,h2,h3,nlevel,Sc,Fr,findex_dict,Fs=1,other_show=False):
    # [c,Bw,fc,i] = Find_wav_findex(x,h,g,h1,h2,h3,nlevel,Sc,Fr,opt2)
    # Sc = -log2(Bw)-1 with Bw the bandwidth of the filter
    # Fr is in [0 .5]
    #
    # -------------------
    # J. Antoni : 12/2004
    # -------------------
    level = np.fix((Sc))+ ((Sc%1) >= 0.5) * (np.log2(3)-1)
    Bw = 2**(-level-1)
    freq_w = np.arange(0,2**(level)) / 2**(level+1) + Bw/2.
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
    kx = findex(c,findex_dict, level)

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

    plt.legend()
    plt.grid(True)
    plt.subplot(2+spec,1,2,sharex=ax1)
    plt.plot(tc,np.abs(c),'k')

    plt.axhline(threshold, c='r')
    
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
        ses_x = f
        ses_y = S[:nfft//2]
        
        plt.plot(ses_x, ses_y,'k')
        #---------------------------------------------------------------------
        plt.title('Fourier transform magnitude of the squared envelope')
        plt.xlabel('frequency [Hz]')
        plt.grid(True)
        
    if other_show:
        plt.show()
    
    return ([c,Bw,fc], [ses_x, ses_y] )

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
    # Calculates the findex K of the complete quinte wavelet packet transform w of signal x, 
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
        if len(bcoeff) ==0:#np.empty(bcoeff):
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
    # print ('findex', findex(c,'findex2'))
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
    #~ v1 = loadmat(r"C:\Users\tlecocq\Documents\Tom's Share\Pack findexogram\Pack findexogram V3\VOIE1.mat")
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

    #-- findexogram data
    x_dict = loadmat('fk_x_test.mat')
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
    
    c = findexgram(x, nlevel, Fs)