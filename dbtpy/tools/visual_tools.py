# -*- coding: utf-8 -*-
"""
Created on Sun May 22 23:02:08 2022

@author: mozhenling
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import pylab as pl 
from dbtpy.findexes.harmonics import harEstimation
from dbtpy.findexes.sigto import sig_real_to_ses
from dbtpy.tools.colorMap import cm_data
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import savemat
import os
parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

def fftspec(fs, sig, normalized=True, oneside=True):
    """
    Inputs:
        fs: sampling frequency
        sig: signal
    Outputs:
        frq_x: frequency values
        frq_y: frequency amplitudes
        normalized: normalized to 0-1
        oneside: return only one side of spectrum

    """
    num_points = len(sig)
    frq_x = np.linspace(0, fs, num_points)  # frequency axis, df=fs/num_points
    frq_y = np.abs(np.fft.fft(sig))
    if normalized:
        frq_y = frq_y / max(frq_y)
    if oneside:
        return frq_x[:num_points//2], frq_y[:num_points//2]
    else:
        return frq_x, frq_y

def fontsizes(fontsize):
    axis_label_fontsize = fontsize + 0.5
    title_fontsize = fontsize + 1
    tick_fontsize = fontsize
    return axis_label_fontsize, title_fontsize,tick_fontsize

def formated_plot(x=None, y=None, xlabel='x', ylabel='y', title=None, fontsize = 10,
                  figdata=None, xlim=None, ylim=None, filename='figdata-formated_plot',
                  save_dir = None, format = 'png', figsize = (8, 4), dpi = 300):

    axis_label_fontsize, title_fontsize,tick_fontsize = fontsizes(fontsize)

    plt.figure(figsize=figsize,dpi=dpi)
    if figdata is not None:
        x, y=figdata['x'], figdata['y']
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if title is not None:
        plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=axis_label_fontsize)
    plt.ylabel(ylabel, fontsize=axis_label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.show()

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        savemat(os.path.join(save_dir, filename+'.'+'mat' ), {'x': x, 'y':y})
        plt.savefig(os.path.join(save_dir, filename+'.'+format ), format=format)

def show_ses_xy(ses_x, ses_y, fs, f_target=None, harN = 3, SSES=True, title='', xlabel = 'Frequency (Hz)',
             ylabel = 'Normalized amplitude',  figsize = (3.5, 1.8), dpi = 144, dev1 = 0.025,
             mark_estimate = True, sig_len_original = None, fre2seq = None,
             fig_save_path= None, fig_format = 'png', fontsize = 8, linewidth = 1, non_text = False ):

    """
    -Applicable for findexgaram
    -Show the square of the squared envelope spectrum
    -We use SES and SSES interchangeably depending on which one can increase SNR
    -By default, we use the SSES

    """
    
    if  fre2seq is not None:
        fre2seq =  fre2seq
    elif sig_len_original is not None:
        fre2seq = sig_len_original / fs
    else:
        fre2seq = len(ses_y) / fs
    
    #-- normalized by the maximum amplitude
    ses_x = np.array(ses_x).squeeze()
    ses_y = np.array(ses_y).squeeze()
    
    sesMax = max(ses_y) 
    ses_y = ses_y /sesMax
    #-- de_label = the lable of the demodulation specturm 
    # you may label it sses instead for sses = True
    (sses_y, label) = (ses_y**2, 'SES' ) if SSES else (ses_y, 'SES') 
    sses_x = ses_x

    plt.figure(figsize=figsize, dpi=dpi)
    plt.rc('font', size = fontsize)
    #-- show calculated fault frequency harmonics as vertical dashed lines
    if f_target is not None:
        
        harColor = ['r', 'r', 'r','m','m', 'm']
        harLine = ['--', '-.', ':','--','-.', ':']
        
        point_num = 10
        targetHarAmp = [np.linspace(0, 1.1, point_num ) for i in range(harN) ]
        targetHar = [[f_target + i*f_target for j in range(point_num)] for i in range(harN) ]
        
        for i, (tar, tarAmp) in enumerate(zip(targetHar, targetHarAmp)):
            plt.plot(tar, tarAmp, harColor[i] + harLine[i],  label ='Har'+str(i+1), linewidth=linewidth + 0.2 )
            # raise ValueError
            plt.ylim([0, 1.1])
            plt.xlim([0, (harN+1) * f_target])
  
    #-- show sses       
    plt.plot( sses_x, sses_y, 'b', label = label, linewidth=linewidth) 
    # show the estimated fault frequency harmonics        
    if mark_estimate:
        fHarAmp, fHarInd = harEstimation(sses_y, f_target, harN = 3, fs = fs,fre2seq=fre2seq,
                                         dev1 = dev1, sig_len_original =sig_len_original) #   dev1 = 0.025
        plt.scatter(x=fHarInd/fre2seq , y=fHarAmp, facecolors='none', edgecolors='r')
      
    if non_text:
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    else:
        plt.xlabel(xlabel, fontsize = fontsize + 0.5)
        plt.ylabel(ylabel, fontsize = fontsize + 0.5)
        plt.title(title, fontsize = fontsize + 1)    
    plt.legend(fontsize = fontsize - 1)
    #-- save the image
    plt.tight_layout()
    if fig_save_path is not None:
        plt.savefig(fig_save_path, format=fig_format)
    plt.show()  
      
def show_ses(sig_real = None, fs=1, f_target=None,  harN = 3, SSES=True, title='', xlabel = 'Frequency (Hz)',
             ylabel = 'Normalized amplitude',  figsize = (3.5, 1.8), dpi = 144,dev1 = 0.025,
             mark_estimate = True, sig_len_original = None, fre2seq = None,
             fig_save_path= None, fig_format = 'png', fontsize = 8, linewidth = 1, non_text = False ):
    """
    -Applicable for optimization based algorithm
    -Show the square of the squared envelope spectrum
    -We use SES and SSES interchangeably depending on which one can increase SNR
    -By default, we use the SSES
    """
    ses = sig_real_to_ses(sig_real)
    Ns = len(ses)
    n = np.arange(Ns)  # start = 0, stop = Ns -1, step = 1
    fn = n * fs / Ns # frequency index, Fs / Ns = frequency resoluton

    ses_x = fn[:Ns//2]
    ses_y = ses[:Ns//2]
    
    show_ses_xy(ses_x, ses_y, fs, f_target, harN , SSES, title, xlabel ,
                 ylabel,figsize, dpi, dev1, mark_estimate, sig_len_original,fre2seq,
                 fig_save_path, fig_format, fontsize, linewidth, non_text)
    
def show_heatmap(Kwav, freq_w, Level_w, nlevel, fs, findex_str=None, figsize = (4, 2), dpi = 144,
             fig_save_path= None, fig_format = 'png', fontsize = 8 ):
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
    #ytick_pos = np.arange(Level_w[0]+0.25, Level_w[-1]+0.75, 0.5)
    ytick_pos = np.arange(0.25, 0.5*(len(Level_w)), 0.5)
    ytick_lab = np.round(np.flip(Level_w * 10))/10
    plt.yticks(ticks = ytick_pos , labels = ytick_lab)
    plt.ylim([Level_w[0],Level_w[-1]])
    plt.xlim([freq_w[0],freq_w[-1]])
    
    # raise ValueError
    plt.xlabel('Frequency (H)', fontsize = fontsize + 0.5)
    plt.ylabel('Level k', fontsize = fontsize + 0.5)
    
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
    
def max_IJ(Kwav):
    # get the maximum value 
    M = np.max(Kwav)
    # get the index of row and col.
    index = np.argmax(Kwav)
    index = np.unravel_index(index, Kwav.shape)
    I = index[0] # row
    J = index[1] # col.
    return [I,J,M]   

def plot_ffts(sig, sig_opt, opt_dict, fs, mfb=None, boundaries=None, figsize = (3.5, 1.8), dpi = 144, 
             blabel=['b0', 'b1'], fig_save_path= None, fig_format = 'png', fontsize = 8, linewidth=1, non_text = False ):
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
    fn = n * fs / Ns # frequency index, Fs / Ns = frequency resoluton
    
    # plt.figure()
    plt.figure(figsize=figsize, dpi=dpi)
    plt.rc('font', size = fontsize)
    
    plt.plot(fn[:Ns//2], sig_fft_amp[:Ns//2], ':', color = 'xkcd:dark sky blue',  
             label = 'Original' , linewidth=linewidth)
    plt.plot(fn[:Ns//2], sig_opt_fft_amp[:Ns//2], 'b', label = 'Optimal' , linewidth=linewidth)
    
    #-- future use
    if mfb is not None:
        mode_len, mode_num = mfb.shape
        style = ['--', '-.', ':']
        for i in range(mode_num): #magenta, light salmon
            #mfb[:Ns:2, i] start=0, stop = Ns, step = 2
            plt.plot(fn[:Ns//2],mfb[:Ns:2, i], style[i], color = 'xkcd:light purple',
                     label = 'filter' + str(i+1), linewidth=linewidth )
            
    if boundaries is not None:
        style_b = ['r--', 'r-.']
        for i in range(len(boundaries)):
            b_x = boundaries[i] * np.ones(10)
            b_y = np.linspace(0, 1, 10)
            plt.plot(b_x, b_y, style_b[i], label =blabel[i], linewidth=linewidth)
    
    plt.rc('font', size = fontsize)

    plt.xlim([0, fn[Ns//2]])
    
    if non_text: 
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    else:
        # ax = plt.gca()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        plt.xlabel('Frequency (Hz)', fontsize = fontsize + 0.5)
        plt.ylabel('Normalized amplitude', fontsize = fontsize + 0.5)
        plt.title(str(opt_dict), fontsize = fontsize + 0.5)
        plt.legend(fontsize = fontsize - 2) 
    
    plt.tight_layout()
    if fig_save_path is not None:
        plt.savefig(fig_save_path, format=fig_format)
    plt.show() 
    
###############################################################################
#------------------------------------------------------------------------------
#-- customized plots
def plot_shape(branches_list, treeID, figsize = (3.5, 1.8), dpi = 144, s_color = 'red', f_color = None,
            xlabel=None, ylabel = None, fig_save_path= None, fig_format = 'png', fontsize = 8, non_text = False):
    
    """
    inputs:
        -branches_list = [x, y, color]
        -treeID, variable name
    """
    
    plt.figure(figsize=figsize, dpi=dpi)
    plt.rc('font', size = fontsize)
    
    b = branches_list
    for i in trange(len(b)):
        #- successful color
        b[i][2] = s_color if b[i][2] == 'orange' and s_color is not None else b[i][2]
        #- failure color
        b[i][2] = f_color if b[i][2] == 'green' and f_color is not None else b[i][2]
 
        plt.plot(b[i][0], b[i][1], color = b[i][2])
    
    plt.gca().invert_yaxis()
    
    if non_text: 
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
    else:
        plt.title('Tree of ' + treeID, fontsize = fontsize + 1)
        if xlabel is not None:
            plt.xlabel(xlabel,fontsize = fontsize + 0.5 )
        else:
            plt.xlabel('Domain of ' + treeID,fontsize = fontsize + 0.5 )
        if ylabel is not None:
            plt.ylabel(ylabel,fontsize = fontsize + 0.5 )
        else:
            plt.ylabel('Depth', fontsize = fontsize + 0.5)
    
    plt.tight_layout()
    if fig_save_path is not None:
        plt.savefig(fig_save_path, format=fig_format)
    plt.show()

def plot_fitness(var_opt_list, figsize = (3.5, 1.8), dpi = 144,
             fig_save_path= None, fig_format = 'png', fontsize = 8):
    """
    plot the fitness
    """
    # if type(var_opt_list) is dict:
    var_opt = [var_list['objValue'] for var_list in var_opt_list]
    # elif type(var_opt_list) is list:
    #     var_opt = var_opt_list
    # else:
    #     raise ValueError
        
    indexes = range(len(var_opt))
    plt.figure(figsize = figsize, dpi = dpi)
    plt.rc('font', size = fontsize)
    
    plt.plot(indexes, var_opt)
    
    plt.title('Fitness Curve', fontsize = fontsize + 1)
    plt.xlabel('Query',fontsize = fontsize + 0.5 )
    plt.ylabel('Fitness', fontsize = fontsize + 0.5)
    
    plt.tight_layout()
    if fig_save_path is not None:
        plt.savefig(fig_save_path, format=fig_format)
    plt.show() 

def plot_sInfo(sInfo_dict, dimName ='x0', yName='sObjValue',  figsize = (3.5, 1.8), dpi = 144,
             fig_save_path= None, fig_format = 'png', fontsize = 8):
    """
    plot the fitness
    -dimName = 'x0' or 'x1' or ... 'xd'
    -yName = 'sObjValue' or 'sVariable'
    """

    y = sInfo_dict[dimName][yName] # 
    x = range(len(y))
    
    plt.figure(figsize = figsize, dpi = dpi)
    plt.rc('font', size = fontsize)
    
    plt.plot(x, y)
    
    plt.title('Successful info. of '+ dimName, fontsize = fontsize + 1)
    plt.xlabel('Query',fontsize = fontsize + 0.5 )
    plt.ylabel(yName, fontsize = fontsize + 0.5)
    
    plt.tight_layout()
    if fig_save_path is not None:
        plt.savefig(fig_save_path, format=fig_format)
    plt.show() 
        
 
def show_time(sig_real = None, title= '', xlabel = 'Time (s)', fs = 1,
             ylabel = 'Normalized amplitude', figsize = (3.5, 1.8), dpi = 144,
             fig_save_path= None, fig_format = 'png', fontsize = 8, linewidth = 1, non_text = False):
    '''
    show time domain waveform only
    '''
    data = sig_real

    
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

     
def show_freq(sig_real=None, fs = 1, title='', xlabel = 'Frequency (Hz)',
             ylabel = 'Normalized amplitude', f_target=None, figsize = (3.5, 1.8), dpi = 144,
             fig_save_path= None, fig_format = 'png', fontsize = 8, linewidth = 1, non_text = False):
    """
    show frequency power spectrum only
    flag = half, 0 to fs // 2
    flag = full, -fs//2 to fs //2
    """
    data = sig_real 

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