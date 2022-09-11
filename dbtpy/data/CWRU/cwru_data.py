# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:09:32 2021

@author: mozhenling
"""

import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import os



class CWRU_Data():
    def __init__(self):
        """
        load Case Western Reserve University data
        
        data structure:
            drive_end_bearing_faults_12kHz_data, about 10 s
            -for Ball = ['118.mat', '119.mat', '120.mat', '121.mat', '187.mat', '224.mat', '225.mat']
                 OR = ['197.mat', '198.mat', '200.mat']
                'XfileName_BA_time'      # array(len, 1) , BA = base plate acceleration 
                'XfileName_DE_time'      # array(len, 1) , DE = drive endacceleration 
                'XfileName_DE_time'      # array(len, 1),  FE = fan endacceleration
                'XfileNameRPM'           # array(1, 1),  sharft speed in rpm
                
            -for IR = ['3001.mat', '3002.mat', '3003.mat', '3004.mat']
                'X056_DE_time'           # '3001.mat'
                'X057_DE_time'           # '3002.mat'
                'X058_DE_time'           # '3003.mat'
                'X059_DE_time'           # '3004.mat'
        
        Ref.:
            Smith, Wade A., and Robert B. Randall. "Rolling element bearing 
            diagnostics using the Case Western Reserve University data: A 
            benchmark study." Mechanical Systems and Signal Processing 64 
            (2015): 100-131.
        """
        self.f_inner = []
        self.f_ball = []
        self.f_outer = []
        self.f_shaft = []
        self.f_cage = []
        
        self.fs = []
        self.resolution = None
        self.fault = ''
        self.sig = []
        self.f_target = []
        
        self.inner_files = ['105.mat', '106.mat', '107.mat', '108.mat',
                            '169.mat', '170.mat', '171.mat', '172.mat',
                            '209.mat', '210.mat', '211.mat', '212.mat',
                            '3001.mat', '3002.mat', '3003.mat', '3004.mat'
                            '056.mat', '057.mat', '058.mat', '059.mat'] # N1 N2 hard
        self.ball_files = ['118.mat', '119.mat', '120.mat', '121.mat',  # N1 N2 hard
                           '185.mat', '186.mat', '187.mat', '188.mat',
                           '222.mat', '223.mat', '224.mat', '225.mat',
                           '3005.mat', '3006.mat', '3007.mat', '3008.mat']
        
        self.outer_files = ['130.mat', '131.mat', '132.mat', '133.mat',
                            '144.mat', '145.mat', '146.mat', '147.mat', 
                            '156.mat', '158.mat', '159.mat', '160.mat',
                            '197.mat', '198.mat', '199.mat', '200.mat',  # N1, N2 hard
                            '234.mat', '235.mat', '235.mat', '237.mat',
                            '246.mat', '247.mat', '248.mat', '249.mat', 
                            '258.mat', '259.mat', '260.mat', '261.mat'] 
        
    def set_f_freq(self, f_shaft_rpm, position = 'DE'):
        """
        set fault frequencies in Hz
        
        inputs:
            f_shaft_rpm # shaft speed in rpm
        """
        if position in ['DE', 'de','drive end', 'Drive End']:
            self.f_shaft = f_shaft_rpm / 60 
            self.f_inner = self.f_shaft * 5.415
            self.f_ball = self.f_shaft * 2.357
            self.f_outer = self.f_shaft * 3.585
            self.f_cage = self.f_shaft * 0.3983
            
        elif position in ['FE', 'fe','fan end', 'Fan End']:
            self.f_shaft = f_shaft_rpm / 60 
            self.f_inner = self.f_shaft * 4.947
            self.f_ball = self.f_shaft * 1.994
            self.f_outer = self.f_shaft * 3.053
            self.f_cage = self.f_shaft * 0.3816   
        else:
            raise ValueError('position of faulty bearing is not correct!')
        
    def load(self, fault_str, fs =12e3 , position = 'DE', resolution = 1, 
             path = r'E:\CityU\CWRU\drive_end_bearing_faults_12kHz_data\N1_N2_hard\IR', 
             sig_detrend = True):
        """  
        inputs:
            -fault_str # name of the file data, e.g. fault_str = '118.mat'
            -resolution # frequency resolution
            -path # file path
            -sig_detrend # detrend the data
        output:
            -sig # fault signal
        """
        self.fault = fault_str
        self.fs = fs
        self.resolution = resolution
 
        #-- load data
        path = os.path.join(path, fault_str)
        N = int(self.fs / resolution) 
        sig_dict = scio.loadmat(path)
        # for now, we just load the DE acceleration 
        f_name, ext=os.path.splitext(fault_str)
        data_name = 'X' + f_name + '_' + position + '_time'
        sig = sig_dict[data_name][0:N, 0]
        if sig_detrend:
            sig = self.detrend(sig)
        # record the load data
        self.sig = sig
        
        #-- set frequencies
        f_shaft_rpm = sig_dict['X' + f_name + 'RPM'].item()
        self.set_f_freq(f_shaft_rpm, position)
        
        if fault_str in self.inner_files:
            self.f_target = self.f_inner
            
        elif fault_str in self.ball_files:
            self.f_target = self.f_ball
            
        elif fault_str in self.outer_files:
            self.f_target = self.f_outer
        else:
            raise NotImplemented
            
        return sig
    
    @property
    def show_time(self):
        '''
        show time domain waveform only
        '''
        data = self.sig
        fs = self.fs
        
        plt.figure()
        Ns = len(data)
        n = np.arange(Ns)  # start = 0, stop = Ns -1, step = 1
        t = n / fs  # time index
        plt.plot(t, data, 'b')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude ')
        plt.title('Time Domain')
        plt.show()
        
    @property   
    def show_freq(self):
        """
        show frequency power spectrum only
        flag = half, 0 to fs // 2
        flag = full, -fs//2 to fs //2
        """
        data = self.sig
        fs = self.fs

        plt.figure()
        Ns = len(data)
        n = np.arange(Ns)  # start = 0, stop = Ns -1, step = 1
        fn = n * fs / Ns # frequency index
        Amp_fn = 2 * abs(np.fft.fft(data)) /Ns

        plt.plot(fn[:len(fn)//2], Amp_fn[:len(fn)//2], 'b') 
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude ')
        plt.title('Frequency Domain')
        plt.show() 
        
    def detrend(self, data, n = 1):
        """
        detrend by polynomail of degree 1
        
        Fit a polynomial p(x) = p[0] * x**deg + ... + p[deg] of degree deg to
        points (x, y). Returns a vector of coefficients p that minimises the
        squared error in the order deg, deg-1, … 0
        
        """
        x = np.arange(0, len(data))
        # Detrend with a 2d order polynomial
        model = np.polyfit(x, data, n )
        predicted = np.polyval(model, x)
        return data - predicted
    
if __name__ == '__main__':
    from dbtpy.findexes.afindex import findex_fun
    from dbtpy.findexes.harmonics import harEstimation
    from dbtpy.tools.visual_tools import show_ses
    from dbtpy.filters.findexgram import findexgram
    #-- load data
    plt.close('all')
    data = CWRU_Data()
    sig_path = r'E:\CityU\CWRU\12k Drive End Bearing Fault Data1\out'
    sig = data.load( fault_str = '156.mat', fs = 12e3, position = 'DE', resolution = 1, path = sig_path)
    
    #---- time and freqeuncy domains of the raw signal
    data.show_time
    data.show_freq
    
    
    #-- for harmonic estimation
    dev1 = 0.05
    #-- ses
    show_ses(sig_real =  sig,fs=data.fs, SSES=True, dpi=300,  dev1 = dev1 ,
              f_target=data.f_target,sig_len_original=len( sig))
    
    
    #-- quick analysis
    findexBase = 'gini'
    findex_dict ={'findex_fun':findex_fun,                      
                      'findex_kwargs':{'findexBase':findexBase, 
                                        'sigD':'env'
                                        }
                      } 
    
    ([M, lev, Bw, fc], [c, ses_x, ses_y], [Kwav, freq_w, Level_w] ) = findexgram(sig = sig, nlevel = 4, 
                                                                                  findex_dict = findex_dict, 
                                                                                  fs = data.fs, 
                                                                                  heatmap_show = True)
    

    