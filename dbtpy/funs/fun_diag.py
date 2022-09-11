# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:50:28 2021

@author: mozhenling
"""
import copy
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

from dbtpy.filters.afilter import filter_fun
from dbtpy.findexes.afindex import findex_fun
from dbtpy.findexes.sigto import sig_real_to_env
#-- import tools
from dbtpy.tools.file_tools import get_time, file_type

###############################################################################
#-------------- the objective function for rotating machinery fault diagnosis
#-------------- It is a maximization problem !!
class Diagnosis():
    def __init__(self, sig, fs=1, filter_kwargs={}, findex_kwargs={}, diag_kwargs={},
                 findex_name=None,filter_name=None):
        """
        fault diagnosis related objective function 
            -sig: the signal
            -fs: the sampling frequency 
            -filter_kwargs: keyword arguments related to the filter
            -findex_kwargs: keyword arguments related to the fault index
            -diag_kwargs: keyword arguments related to this diagnosis class
        """

        self.sig = sig
        self.fs = fs
        
        self.filter_kwargs = filter_kwargs
        self.findex_kwargs = findex_kwargs
        self.diag_kwargs = diag_kwargs # you may pass other key word arguments here
        #-- to name the fault index that will be shown to us 
        if findex_name is None:
            if findex_kwargs['sigD'] is not None:
                self.findex_name = findex_kwargs['findexBase'] + '_' + findex_kwargs['sigD'] 
            else:
                self.findex_name = findex_kwargs['findexBase'] 
        else:
            self.findex_name = findex_name 
        #-- to name the filter that will be shown to us    
        if filter_name is None:
            self.filter_name = filter_kwargs['filterBase'] 
        else:
            self.filter_name = filter_name
        self.findex_opt = [] # record the optimal fault index
        self.sig_opt = []    # record the optimal signal 
        self.sig_opt_mfb = []        # for meyer 
        self.full_opt_info = [] # record the optimal variabes and the coressponding obj. values
        self.full_try_info = [] # record the tried variabes and the coressponding obj. values
        
    def pso_vmd(self, variables):
        """
        input:
            variables: alpha = variables[0], K = round(variables[1])
        ideas:
            -the fitness is the entropy and the kurtosis
            -minimize entropy to determine the decomposition number k and compactness factor alpha 
            -maximize kurtosis to select a model after the decomposition
        ref.:
            Wang, Xian-Bo, Zhi-Xin Yang, and Xiao-An Yan. "Novel particle swarm 
            optimization-based variational mode decomposition method for the 
            fault diagnosis of complex rotating machinery." IEEE/ASME Transactions
            on Mechatronics 23.1 (2017): 68-79.
        """
        alpha = variables[0]
        K = round(variables[1])
        
        mode, _, _ = filter_fun(self.sig, alpha=alpha, K=K, **self.filter_kwargs)
        # the envelope moduel, which will be used to calcualte envelope energy entropy
        mode_env_list = [sig_real_to_env(mode[i,:]) for i in range(K)]
        # obtain the envelope energy entropy
        findex = findex_fun(mode_env_list, **self.findex_kwargs)
        
        #-- get the optimal sig of each call
        mode_kurt_list = [findex_fun(mode[i,:], 'kurt', 'se') for i in range(K)]
        mode_kurt_max = max(mode_kurt_list)
        ind_max = mode_kurt_list.index(mode_kurt_max)
        
        # record the optimal index 
        if self.findex_opt==[]: 
            self.findex_opt = [[findex, mode_kurt_max]]
            self.full_opt_info.append({'alpha': alpha, 'K': K, 'entropy_min':-1*findex, 'kurt_max':mode_kurt_max})
        elif  findex > self.findex_opt[-1][0]:
            self.findex_opt.append([findex, mode_kurt_max])
            self.sig_opt = mode[ind_max,:] 
            self.full_opt_info.append({'alpha': alpha, 'K': K, 'entropy_min':-1*findex, 'kurt_max':mode_kurt_max})
        else:
            self.findex_opt.append(self.findex_opt[-1])
            self.full_opt_info.append(self.full_opt_info[-1])
        self.full_try_info.append({'alpha': alpha, 'K': K, 'entropy_min':-1*findex, 'kurt_max':mode_kurt_max})
        
        return findex
        
    def ovmd_alike(self, variables):
        """
        Optimized variantional mode decomposition and its vairants by maximizing the fualt index
        input:
            variables: alpha = variables[0], K = round(variables[1])      
        outputs:
            -the largest fault index
        """
        alpha = variables[0]
        K = round(variables[1])

        mode, _, _ = filter_fun(self.sig, alpha=alpha, K=K, **self.filter_kwargs)
         
        # the higher the fault index, the more likely a fault has happened
        findex_list = [ findex_fun( mode[i,:], **self.findex_kwargs ) for i in range(K) ] 
        findex_max = max( findex_list ) 
        ind_max = findex_list.index(findex_max)
        # save the latest fualt index 
        # record the optimal index 
        if self.findex_opt==[]: 
            self.findex_opt = [findex_max]
            self.sig_opt = mode[ind_max,:]
            self.full_opt_info.append({'alpha': alpha, 'K': K, self.findex_name:findex_max})
        elif findex_max > self.findex_opt[-1]:
            self.findex_opt.append(findex_max)
            self.sig_opt = mode[ind_max,:]
            self.full_opt_info.append({'alpha': alpha, 'K': K, self.findex_name:findex_max})
        else:
            self.findex_opt.append(self.findex_opt[-1])
            self.full_opt_info.append(self.full_opt_info[-1])        
        self.full_try_info.append({'alpha': alpha, 'K': K, self.findex_name:findex_max})
        return findex_max 
    
    def meyer3_others(self, b):
        """
        The three meyer wavelet filters based objective funtion for fault diagnosis
        using other optimization algorithms such as those from mealpy 
        inputs:
            -b     # the boundaries for the filter bank
            -minB  # the minimum band width requirements (via diag_kwargs['minB'])
            
        outputs:
            -the largest fault index 
        """

        #-- it is convinient if other optimization algorithm only allows one argument
        minB =self.diag_kwargs['minB'] 
        filter_num = 3
        # check the value of bounds regarding the bandwidth constraint
        if abs(b[1]-b[0]) > minB:
            b[0], b[1] = min(b), max(b)
        else:
            b[0] = np.mean(b)
            b[1] = b[0] + minB
        
        #-- convert to sequence index
        freq2seq = len(self.sig) / self.fs 
        b = np.array(b) * freq2seq
        
        mode, mfb ,boundaries  = filter_fun(self.sig,  boundaries=b, **self.filter_kwargs)
        # the higher the fault index, the more likely a fault has happened
        
        #--- calcualte findexes of three modes
        findex_list= [ findex_fun( mode[:,i], **self.findex_kwargs ) for i in range(filter_num) ] 
        findex_max = max( findex_list ) 
        ind_max = findex_list.index(findex_max)
        
        # record the optimal index 
        if self.findex_opt == []: # if it is empty
            self.findex_opt = [findex_max]
            self.sig_opt = mode[:,ind_max]
            self.full_opt_info.append({'b1': b[0], 'b2': b[1], self.findex_name:findex_max}) # findex
        elif findex_max > self.findex_opt[-1]:
            self.findex_opt.append(findex_max)
            self.sig_opt = mode[:,ind_max]
            self.full_opt_info.append({'b1': b[0], 'b2': b[1], self.findex_name:findex_max})
        else:
            self.findex_opt.append(self.findex_opt[-1])
            self.full_opt_info.append(self.full_opt_info[-1])
        
        self.full_try_info.append({'b1': b[0], 'b2': b[1], self.findex_name:findex_max})
        
        return findex_max
    
    def meyer3_dbt(self, b):
        """
        The three meyer wavelet filters based objective funtion for fault diagnosis
        using dbtree optimization algorithms 
        inputs:
            -b     # the boundaries for the filter bank
            -minB  # the minimum band width requirements (via obj_kwargs['minB'])
            
        outputs:
            -the largest fault index 
        """
        #-- it is convinient if other optimization algorithm only allows one argument
        minB =self.diag_kwargs['minB'] 
        
        filter_num = 3
        # check the value of bounds regarding the bandwidth constraint
        if abs(b[1]-b[0]) > minB:
            b[0], b[1] = min(b), max(b)
        else:
            b[0] = np.mean(b)
            b[1] = b[0] + minB
        
        #-- convert to sequence index
        freq2seq = len(self.sig) / self.fs # np.pi / (self.fs / 2)
        b = np.array(b) * freq2seq
                 
        mode, mfb ,boundaries  = filter_fun(self.sig, boundaries=b, **self.filter_kwargs)
        # the higher the fault index, the more likely a fault has happened
        
        #--- calcualte findexes of three modes
        findex_list= [ findex_fun( mode[:,i], **self.findex_kwargs ) for i in range(filter_num) ] 
        findex_max = max( findex_list ) 
        ind_max = findex_list.index(findex_max)
        
        # record the optimal index 
        if self.findex_opt == []: # if it is empty
            self.findex_opt = findex_max
            self.sig_opt = mode[:,ind_max]
            self.sig_opt_mfb = mfb
        elif findex_max > self.findex_opt:
            self.findex_opt = findex_max
            self.sig_opt = mode[:,ind_max]
            self.sig_opt_mfb = mfb
        else:
            pass     
        return findex_max 
    
    def to_2pi(self, freq):
        """
        map a frequency in [0, fs/2] to the corresponding frequency in [0, 2pi]
        where 2pi = fs, pi = fs / 2
        """
        return freq * 2 * np.pi / self.fs
        
    def to_fs(self, omega):
        """
        map a frequency in [0, pi] to the corresponding frequency in [0, fs/2]
        where 2pi = fs, pi = fs / 2
        """
        return omega * self.fs / (2 * np.pi)
    
    def show_time(self, sig_real = None, title= '', xlabel = 'Time (s)',
                 ylabel = 'Normalized amplitude', figsize = (3.5, 1.8), dpi = 144,
                 fig_save_path= None, fig_format = 'png', fontsize = 8, linewidth = 1, non_text = False):
        '''
        show time domain waveform only
        '''
        data = sig_real if sig_real is not None else self.sig_opt
        fs = self.fs
        
        plt.figure(figsize=figsize, dpi=dpi)
        plt.rc('font', size = fontsize)
        
        data = data / max(abs(data))
        
        Ns = len(data)
        n = np.arange(Ns)  # start = 0, stop = Ns -1, step = 1
        t = n / fs  # time index
        plt.plot(t, data, color = 'xkcd:dark sky blue', linewidth = linewidth)
        
        if non_text:
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        else:
            plt.xlabel(xlabel, fontsize = fontsize + 0.5)
            plt.ylabel(ylabel, fontsize = fontsize + 0.5)
            plt.title(title, fontsize = fontsize + 1)
        
        plt.tight_layout()
        if fig_save_path is not None:
            plt.savefig(fig_save_path, format=fig_format)
        plt.show() 
    
         
    def show_freq(self, sig_real=None, fs = None, title='', xlabel = 'Frequency (Hz)',
                 ylabel = 'Normalized amplitude', f_target=None, figsize = (3.5, 1.8), dpi = 144,
                 fig_save_path= None, fig_format = 'png', fontsize = 8, linewidth = 1, non_text = False):
        """
        show frequency power spectrum only
        flag = half, 0 to fs // 2
        flag = full, -fs//2 to fs //2
        """
        data = sig_real if sig_real is not None else self.sig_opt
        fs = fs if fs is not None else self.fs 

        plt.figure(figsize=figsize, dpi=dpi)
        plt.rc('font', size = fontsize)
        
        Ns = len(data)
        n = np.arange(Ns)  # start = 0, stop = Ns -1, step = 1
        fn = n * fs / Ns # frequency index
        if np.iscomplexobj(data):
            Amp_fn = abs(data)
        else:
            Amp_fn = 2 * abs(np.fft.fft(data)) /Ns
        
        Amp_fn = Amp_fn / max(Amp_fn) 
        
        plt.plot(fn[:len(fn)//2], Amp_fn[:len(fn)//2], color = 'xkcd:dark sky blue', linewidth = linewidth) 
        
        if non_text:
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        else:
            plt.xlabel(xlabel, fontsize = fontsize + 0.5)
            plt.ylabel(ylabel, fontsize = fontsize + 0.5)
            plt.title(title, fontsize = fontsize + 1)

        plt.tight_layout()
        if fig_save_path is not None:
            plt.savefig(fig_save_path, format=fig_format)
        plt.show() 

    
    def show_sses(self, sig_real = None, f_target=None, SSES=True, title='', xlabel = 'Frequency (Hz)',
                 ylabel = 'Normalized amplitude',  figsize = (3.5, 1.8), dpi = 144,
                 fig_save_path= None, fig_format = 'png', fontsize = 8, linewidth = 1, non_text = False ):
        """
        show the square of the squared envelope spectrum
        """
        sig = sig_real if sig_real is not None else self.sig_opt
        ses = sig_real_to_ses(sig)
        #-- normalized by the maximum amplitude
        sesMax = max(ses)
        ses = ses / sesMax
        (sses, label)= (ses**2, 'SES') if SSES else (ses, 'SES') # you may use sses instead

        # plt.figure()
        plt.figure(figsize=figsize, dpi=dpi)
        
        plt.rc('font', size = fontsize)
        
        if f_target is not None:
            harN = 5
            harColor = ['r', 'r', 'r', 'm', 'm']
            harLine = ['--', '-.', ':', '--', '-.']
            point_num = 10
            targetHarAmp = [np.linspace(0, 1.1, point_num ) for i in range(harN) ]
            targetHar = [[f_target + i*f_target for j in range(point_num)] for i in range(harN) ]
            
            for i, (tar, tarAmp) in enumerate(zip(targetHar, targetHarAmp)):
                plt.plot(tar, tarAmp, harColor[i] + harLine[i],  label ='Har'+str(i+1), linewidth=linewidth + 0.3)
                # raise ValueError
                plt.ylim([0, 1.1])
                plt.xlim([0,  7* f_target]) # (harN + 1)
            
        
        Ns = len(sses)
        n = np.arange(Ns)  # start = 0, stop = Ns -1, step = 1
        fn = n * self.fs / Ns # frequency index, Fs / Ns = frequency resoluton
        plt.plot(fn[:Ns//2], sses[:Ns//2], 'b', label = label , linewidth=linewidth)
        
        
        if non_text:
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        else:
            plt.xlabel(xlabel, fontsize = fontsize + 0.5)
            plt.ylabel(ylabel, fontsize = fontsize + 0.5)
            plt.title(title, fontsize = fontsize + 1)    
        plt.legend(fontsize = fontsize - 1)
        
        plt.tight_layout()
        if fig_save_path is not None:
            plt.savefig(fig_save_path, format=fig_format)
        plt.show()  
 
    def plot_ffts(self, sig, sig_opt, opt_dict, mfb=None, boundaries=None, figsize = (3.5, 1.8), dpi = 144,
                 fig_save_path= None, fig_format = 'png', fontsize = 8, linewidth=1, non_text = False ):
        """
        plot the fft of the original signal and the filtered optimal signal
        
        inputs:
            -sig         # the original
            -sig_opt     # the optimal sigal
            -vars_opt    # the optimal decision variables
            -ffindex  # name of the fault index
            -findex_opt  # value of the optimal fault index
        """
        
        sig_fft_amp = abs(np.fft.fft(sig))
        #-- normalized by the maximum amplitude
        amp_max = max(sig_fft_amp)
        sig_fft_amp = sig_fft_amp / amp_max 
        sig_opt_fft_amp = abs(np.fft.fft(sig_opt)) / amp_max 
        
        Ns = len(sig_fft_amp)
        n = np.arange(Ns)  # start = 0, stop = Ns -1, step = 1
        fn = n * self.fs / Ns # frequency index, Fs / Ns = frequency resoluton
        
        # plt.figure()
        plt.figure(figsize=figsize, dpi=dpi)
        
        plt.plot(fn[:Ns//2], sig_fft_amp[:Ns//2], ':', color = 'xkcd:dark sky blue',  label = 'Original' , linewidth=linewidth)
        plt.plot(fn[:Ns//2], sig_opt_fft_amp[:Ns//2], 'b', label = 'Optimal' , linewidth=linewidth)
        
        #-- future use
        if mfb is not None:
            mode_len, mode_num = mfb.shape
            style = ['--', '-.', ':']
            for i in range(mode_num): #magenta, light salmon
                plt.plot(fn[:Ns//2],mfb[:Ns:2, i], style[i], color = 'xkcd:light purple', label = 'filter' + str(i+1), linewidth=linewidth )
                
        if boundaries is not None:
            style_b = ['r--', 'r-.']
            for i in range(len(boundaries)):
                b_x = boundaries[i] * np.ones(10)
                b_y = np.linspace(0, 1, 10)
                plt.plot(b_x, b_y, style_b[i], label = 'b' + str(i+1), linewidth=linewidth)
        
        plt.rc('font', size = fontsize)

        plt.xlim([0, fn[Ns//2]])
        
        if non_text: 
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        else:
            plt.xlabel('Frequency (Hz)', fontsize = fontsize + 0.5)
            plt.ylabel('Normalized amplitude', fontsize = fontsize + 0.5)
            plt.title(str(opt_dict), fontsize = fontsize + 0.5)
            plt.legend(fontsize = fontsize - 2) 
        
        plt.tight_layout()
        if fig_save_path is not None:
            plt.savefig(fig_save_path, format=fig_format)
        plt.show()   
 
if __name__ == '__main__':
    a = np.zeros((1000))
