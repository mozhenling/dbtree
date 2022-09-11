# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 20:16:24 2020

@author: mozhenling
"""
import torch 
import torch.fft
import numpy as np
import scipy.io as sio
# #-- suzhou university data
# Ns = 10000  # sampling pionts
# fs = 10000  # sampling frequency
# data_dic = sio.loadmat('su0.2mm0kN-InnerRollingOuter-4#-Fs10k-Ns40960.mat')
# data = data_dic['data'][0:Ns,0]

# # signal[batchsize, channel, signal]
# batch = np.arange(0, 2 )
# channel = np.arange(0, 1 )
# signal = data
# # temp_ini = np.zeros([len(batch), len(channel), len(signal)])
# waveforms = torch.from_numpy(np.zeros([len(batch), len(channel), len(signal)]))
# [batch_size, channel_size, signal_size] = waveforms.shape
# # load data to batch waveforms
# for i in batch:
#     for j in channel:
#         waveforms.numpy()[i, j, :] = signal

# max_iter_num=300
# max_mode_num=1
# Alpha=torch.from_numpy(np.array([100, 100, 100]))
# omega_init=torch.zeros(batch_size, channel_size, max_mode_num)
# tau=torch.from_numpy(np.array([0.]))
# tol=torch.from_numpy(np.array([1e-7]))
# device = 'cpu'

def SVMD_VKnet_torch(waveforms, Alpha, omega_init, tau, tol, max_iter_num, max_mode_num, device):
    '''
    Torch version of SVMD without mirroring
    Author of python version : MO Zhenling
    E-mail: zhenlmo2-c@my.city.edu.hk
    ---------------------------------------------------------------------------
    Inputs:
    ---------------------
      signal       - the time domain input signal
      Alpha        - compactness of mode constraint
      omega_int    - initial guess of mode center-frequency 
                    # 1) omega_int = np.array([f1, f2, f3, ...]))/fs
                         user-defined normalized frequency
                    # 2) omega_int = 1 initilize uniformly
                    # 3) omega_int = 2 initilize randomly 
                    # otherwise: initilize as zeros
      fs           - sampling frequency
      tau          - time-step of the dual ascent. set it to 0 in the presence of
                   high level of noise.
      tol          - tolerance of convergence criterion; typically around 1e-6
      max_iter_num - maximum iteration number for each mode extraction
      max_mode_num - maximum mode number desired
      noise_power  - white noise power (future parameter)
    
    Defualts:
      Alpha=torch.from_numpy(np.array([100, 100, 100])).squeeze()
      omega_int=torch.zeros(Alpha.shape)
      tau=torch.from_numpy(np.array([0.])).squeeze()
      tol=torch.from_numpy(np.array([1e-7])).squeeze()
      max_iter_num=300
      max_mode_num=3
    ---------------------------------------------------------------------------
    Outputs:
    ---------------------
      modes       - modes obtained by svmdpy and presented in time domain
      modes_fn    - modes obtained by svmdpy and presented in frequency domain
      modes_fc    - modes centeral frequencies
      omega      - history of estimated mode center-frequency(future parameter)
    '''
    # ------------------ Part 1: Start initializing----------------------------
    # Time Domain 0 to T (of mirrored signal
    [batch_size, channel_size, signal_size] = waveforms.shape
    T = signal_size
    t = torch.arange(1,T+1)/T  
    # fs = 1./T
    # Spectral Domain discretization
    omega_axis = (t-0.5-(1/T)).to(device) 
   #-------- FFT of signal(and Hilbert transform concept=making it one-sided)
   # for real inputs, the negative frequencies are redundant
   # rfft contains only the positive frequencies below the Nyquist frequency
    waveforms = waveforms - torch.mean(waveforms, dim = 2, keepdim= True) # remove DC part
    f_hat_plus_ = torch.fft.rfft(waveforms, dim = 2) 
    f_hat_negt_ = torch.zeros( batch_size, channel_size, signal_size - f_hat_plus_.shape[2], dtype = torch.cfloat).to(device)
    # keeping only the postive spectrum while reserving negative side axis for future use (for complex signal)
    f_hat_plus = torch.cat( (f_hat_negt_, f_hat_plus_), dim = 2)
    # desired mode in frequency domain during estimation
    # if running out of memory, use cfloat instead of cdouble
    mode_hat_d = torch.zeros(max_mode_num, batch_size, channel_size, signal_size, dtype = torch.cfloat).to(device) 
    # mode central frequency   
    # mode_fc = np.zeros([max_iter_num,max_mode_num])
    modes_fc = torch.zeros( batch_size, channel_size, max_mode_num).to(device)
    # outr loop counter 
    n_outer = 0
    # if running out of memory, use cfloat instead of cdouble
    sum_omega_old = torch.zeros(max_mode_num + 1, batch_size, channel_size, signal_size,dtype = torch.float).to(device)  # sum of the old central frequency ralated parts
    # if running out of memory, use cfloat instead of cdouble
    sum_mode_hat_d = torch.zeros(batch_size, channel_size, signal_size,dtype = torch.cfloat).to(device)
    # initialization of center frequency of each mode
    # omega_int_ = init_omega_central(omega_int)
    # omega_init_ = omega_init 
    # ------------------ Part 2: Main loop for iterative updates---------------
    while(n_outer < max_mode_num):
        # inner loop conter 
        n_inner = 0
        # update step
        udiff = tol + tol
        #omediff = tol+np.spacing(1)
       # Initializing omega_d for the next run
        omega_d = torch.zeros(batch_size, channel_size, max_iter_num).to(device)
        #omega_d[1] = omega_d[n_inner];
        omega_d[:, :, 0] = omega_init[:,:, n_outer]
        # dual variables vector
        #np.zeros([Niter, len(freqs)], dtype = complex)
         # if running out of memory, use cfloat instead of cdouble
        lambda_hat = torch.zeros(max_iter_num, batch_size, channel_size, signal_size,dtype = torch.cfloat).to(device)
        # keeping changes of mode spectrum
         # if running out of memory, use cfloat instead of cdouble
        u_hat_d = torch.zeros(max_iter_num, batch_size, channel_size, signal_size,dtype = torch.cfloat).to(device)
        # if running out of memory, use .float() instead of .double()
        omega_axis_dot = (omega_axis + torch.zeros(u_hat_d.shape).to(device)).float().to(device)
        #-------------- inner loop for mode extractions
        while(udiff > tol and  n_inner < max_iter_num -1 ):
        #while(  n_inner < max_iter_num -1 ):    
            #----------- update u_hat_d
            u_hat_d[n_inner+1,:, :, :] = (f_hat_plus + (u_hat_d[n_inner,:, :,:] * \
            (Alpha[n_outer]**2) * (omega_axis - omega_d[:,:,n_inner]).view(-1,channel_size,signal_size)**4) + lambda_hat[n_inner,:, :,:] / 2) \
            /((1 + (Alpha[n_outer]**2) * (omega_axis - omega_d[:,:,n_inner])**4).view(-1,channel_size,signal_size) * \
            (1 + 2 * Alpha[n_outer] * (omega_axis - omega_d[:,:,n_inner]).view(-1,channel_size,signal_size)**2 + \
            sum_omega_old[n_outer, :, :,:]))            
            #-----------update omega_d
            # omega_axis_temp = omega_axis + torch.zeros_like(u_hat_d)
            omega_d[:,:,n_inner+1] = (torch.tensordot(omega_axis_dot[n_inner+1,:, :, T//2:T], 
                                                      (torch.abs(u_hat_d[n_inner+1,:, :, T//2:T])**2), dims = 3) \
            /torch.sum(torch.abs(u_hat_d[n_inner+1, :, :,T//2:T])**2))#.view(batch_size, channel_size,-1)      
            #------ update lambda_hat
            lambda_hat[n_inner+1,:, :,:] = lambda_hat[n_inner,:, :,:] +  tau *  \
                ( f_hat_plus - ( u_hat_d[n_inner+1,:, :,:] + \
                ( (Alpha[n_outer]**2 * (omega_axis - omega_d[:,:,n_inner+1]).view(-1,channel_size,signal_size)** 4) * \
                (f_hat_plus - u_hat_d[n_inner+1,:, :,:] - sum_mode_hat_d + lambda_hat[n_inner,:, :,:]/2)- \
                sum_mode_hat_d / (1+2*Alpha[n_outer]**2 * (omega_axis -omega_d[:,:,n_inner+1]).view(-1,channel_size,signal_size)**4)) + \
                sum_mode_hat_d ) ) 
            n_inner = n_inner + 1
            udiff = torch.zeros(1).to(device)
            udiff = udiff + 1/T * torch.tensordot((u_hat_d[n_inner,:, :,:]-u_hat_d[n_inner-1,:, :,:]),
                                                  torch.conj((u_hat_d[n_inner,:, :,:]- u_hat_d[n_inner-1,:, :,:])),dims = 3)           
            udiff = udiff /torch.tensordot(u_hat_d[n_inner,:, :,:], torch.conj(u_hat_d[n_inner,:, :,:]), dims = 3)
            udiff = torch.real(udiff)
            #omega_d = abs(omega_d[n_inner] - omega_d[n_inner - 1]) / abs(omega_d[n_inner])
        mode_hat_d[n_outer,:,:,:]= u_hat_d[n_inner,:, :,:]
        sum_mode_hat_d = sum_mode_hat_d + u_hat_d[n_inner,:, :,:]
        sum_omega_old[n_outer+1,:, :,:]= sum_omega_old[n_outer,:, :,:] + 1 / ((Alpha[n_outer]**2) * (omega_axis - omega_d[:,:,n_inner]).view(-1,channel_size,signal_size)**4)
        # mode_fc[:, n_outer] = omega_d.reshape(max_iter_num)
        modes_fc[:,:,n_outer] = omega_d[:,:,n_inner]
        n_outer = n_outer + 1
        
    # ------------------ Part 3: Signal Reconstruction-------------------------
    # modes : [max_mode_num, batch_size, channel_size, signal_size]
    # sort the mode according to the central frequency
    # modes_fc, indixes = torch.sort(modes_fc, dim=2)
    # indixes = indixes.permute(2,1,0)
    # mode_hat_d = mode_hat_d[indixes,:]
    
    modes = torch.fft.irfft(mode_hat_d[:,:,:,f_hat_negt_.shape[2]:], dim = 3)
    modes_fn = mode_hat_d # mode in frequency domain 
    
    return modes, modes_fn, modes_fc, omega_axis, sum_omega_old,f_hat_plus, f_hat_negt_   
    
def filter_VKnet_torch(Alpha, omega_axis, modes_fc, sum_omega_old, f_hat_plus, f_hat_negt_, device):
    """
    used to construct variational filter from SVMD without mirroring
    alpha = [alpha_1, alpha_2, alpha_3]
    omega_axis: (signal_size)
    modes_fc: (batch_size, channel_size, fc_size) fc_size = modes_size
    sum_omega_old: (max_mode_num + 1, batch_size, channel_size, signal_size,dtype = torch.cdouble)
    f_hat_plus: (batch_size, channel_size, signal_size), one-side is zeros
    f_hat_negt_: nagtive counterpart of f_hat_plus, all zeros, for future use
    """
    [max_mode_num, batch_size, channel_size, signal_size] = sum_omega_old.shape
    max_mode_num = max_mode_num - 1
     # if running out of memory, use cfloat instead of cdouble
    filter_hat_d = torch.zeros(max_mode_num, batch_size, channel_size, signal_size, dtype = torch.cfloat).to(device) 
    for i in range(max_mode_num):
        g_omega = (Alpha[i]**2) * (omega_axis - modes_fc[:,:,i]).view(-1,channel_size,signal_size)**4 
        # denomenator
        h_omega =((1 + (Alpha[i]**2) * (omega_axis - modes_fc[:,:,i]).view(-1,channel_size,signal_size)**4) * \
               (1 + 2 * Alpha[i] * (omega_axis - modes_fc[:,:,i]).view(-1,channel_size,signal_size)**2 + \
                 sum_omega_old[i,:,:,:]))
        filter_hat_d[i,:,:,:] = f_hat_plus / (h_omega  - g_omega ) # treate as one mode 
    
    mode_filters = torch.fft.irfft(filter_hat_d[:,:,:,f_hat_negt_.shape[2]:], dim = 3)
    mode_filters_fn = filter_hat_d / len(omega_axis)
    # mode_filter_ng = filter_hat_d.clone().detach()
    # mode_filters_fn = (torch.flip(mode_filter_ng, dims = [3]) + filter_hat_d) / len(omega_axis)
     
    return mode_filters, mode_filters_fn #torch.abs(mode_filters_fn) # return the one-side abs spectrum   

def filter_VKnet_time_torch(Alpha, omega_axis, modes_fc, sum_omega_old, f_hat_plus, f_hat_negt_, device):
    """
    used to construct variational filter from SVMD without mirroring
    alpha = [alpha_1, alpha_2, alpha_3]
    omega_axis: (signal_size)
    modes_fc: (batch_size, channel_size, fc_size) fc_size = modes_size
    sum_omega_old: (max_mode_num + 1, batch_size, channel_size, signal_size,dtype = torch.cdouble)
    f_hat_plus: (batch_size, channel_size, signal_size), one-side is zeros
    f_hat_negt_: nagtive counterpart of f_hat_plus, all zeros, for future use
    """
    [max_mode_num, batch_size, channel_size, signal_size] = sum_omega_old.shape
    max_mode_num = max_mode_num - 1
     # if running out of memory, use cfloat instead of cdouble
    filter_hat_d = torch.zeros(max_mode_num, batch_size, channel_size, signal_size, dtype = torch.cfloat).to(device) 
    for i in range(max_mode_num):
        g_omega = (Alpha[i]**2) * (omega_axis - modes_fc[:,:,i]).view(-1,channel_size,signal_size)**4 
        # denomenator
        h_omega =((1 + (Alpha[i]**2) * (omega_axis - modes_fc[:,:,i]).view(-1,channel_size,signal_size)**4) * \
               (1 + 2 * Alpha[i] * (omega_axis - modes_fc[:,:,i]).view(-1,channel_size,signal_size)**2 + \
                 sum_omega_old[i,:,:,:]))
        filter_hat_d[i,:,:,:] = 1 / (h_omega  - g_omega ) # treate as one mode 
    
    mode_filters = torch.fft.irfft(filter_hat_d[:,:,:,f_hat_negt_.shape[2]:], dim = 3)
    # mode_filters_fn = filter_hat_d / len(omega_axis)
    # mode_filter_ng = filter_hat_d.clone().detach()
    # mode_filters_fn = (torch.flip(mode_filter_ng, dims = [3]) + filter_hat_d) / len(omega_axis)
     
    return mode_filters#, mode_filters_fn #torch.abs(mode_filters_fn) # return the one-side abs spectrum  

def filter_VKnet_real_torch(Alpha, omega_axis, modes_fc, sum_omega_old, f_hat_plus, f_hat_negt_, device):
    """
    used to construct variational filter from SVMD without mirroring
    alpha = [alpha_1, alpha_2, alpha_3]
    omega_axis: (signal_size)
    modes_fc: (batch_size, channel_size, mode_num) fc_size = modes_size
    sum_omega_old: (max_mode_num + 1, batch_size, channel_size, signal_size,dtype = torch.cdouble)
    f_hat_plus: (batch_size, channel_size, signal_size), one-side is zeros
    f_hat_negt_: nagtive counterpart of f_hat_plus, all zeros, for future use
    """
    [max_mode_num, batch_size, channel_size, signal_size] = sum_omega_old.shape
    max_mode_num = max_mode_num - 1 # remove an extra non-meaningfule mode
    sum_omega_old=sum_omega_old[:max_mode_num,:,:,:].view(max_mode_num, batch_size, channel_size, signal_size)
    # separate real and imaginary parts
    # sum_omega_old = torch.complex(sum_omega_old, torch.zeros_like(sum_omega_old))
    # sum_omega_old_real = torch.real(sum_omega_old)
    f_hat_plus_real = torch.real(f_hat_plus)
    # sum_omega_old_imag = torch.imag(sum_omega_old)
    f_hat_plus_imag = torch.imag(f_hat_plus)
    # # f_hat_negt_ = torch.real(f_hat_negt_)
    
     # if running out of memory, use cfloat instead of cdouble
     # store the power spectrum
    mode_filter_ps = torch.zeros(max_mode_num, batch_size, channel_size, signal_size, dtype = torch.float).to(device)
    modes_fc=modes_fc.permute(2,0,1) # (mode_num, batch_size, channel_size) # channel = 1
    modes_fc=modes_fc.unsqueeze(-1) # (mode_num, batch_size, channel_size, 1)
    
    Alpha = (Alpha + torch.zeros_like(mode_filter_ps, dtype = torch.float).permute(1,2,3,0)) #  batch_size, channel_size, signal_size,max_mode_num
    Alpha=Alpha.permute(3,0,1,2)
    g_omega = ((Alpha**2) * (omega_axis - modes_fc)**4).view(max_mode_num, batch_size, channel_size, signal_size)
    # denomenator
    h_omega=((1 + (Alpha**2) * (omega_axis - modes_fc).view(max_mode_num, batch_size, channel_size, signal_size)**4) * \
           (1 + 2 * Alpha * (omega_axis - modes_fc).view(max_mode_num, batch_size, channel_size, signal_size)**2 + \
             sum_omega_old))
        
    mode_filter_ps=  torch.sqrt(torch.abs(f_hat_plus_real / (h_omega - g_omega))**2  + \
        torch.abs(f_hat_plus_imag / (h_omega - g_omega))**2 ) / len(omega_axis)

     
    return  mode_filter_ps# mode_filter_ps #torch.abs(mode_filters_fn) # return the one-side abs spectrum   

def get_numpy_copy(data):
    return data.clone().detach().cpu().numpy()


def omega_d_init_(batch_size, channel_size, out_channels, ini_type = 0, device = 'cuda'):
    # initialize as zeros
    if ini_type == 0:
        return torch.zeros(batch_size, channel_size, out_channels).to(device)
    # initializer uniformly
    if ini_type == 1:
        init_temp = torch.zeros(batch_size, channel_size, out_channels).to(device)
        temp = torch.zeros(out_channels).to(device)
        for i in range(out_channels):
            temp[i] = ((0.5/out_channels)*(i) + (0.5/out_channels)*(i+1)) / 2
        return (init_temp + temp).to(device)
    # initialize randomly in [0, 0.5], sort the last dimension in ascending order by value
    if ini_type == 2:
        ome_temp, ind = torch.sort(0.5 * torch.rand(batch_size, channel_size, out_channels).to(device), dim=2)
        return ome_temp
    else:
        return torch.zeros(batch_size, channel_size, out_channels).to(device)

# def init_omega_central(fs, omega_init,max_iter_num, max_mode_num):
#     if torch.is_tensor(omega_init):
#         return omega_init
#     else:
#         if isinstance(omega_init, list) or isinstance(omega_init, np.ndarray):
#             omega_init_ = omega_init
#         elif isinstance(omega_init, int):
#             if  omega_init == 0:
#                 omega_init_ = torch.zeros(max_iter_num)                
#             if omega_init == 1:
#                  omega_init_ = torch.zeros(max_iter_num)
#                  # initialize uniformly
#                  for i in range(max_mode_num): 
#                      omega_init_[i] = (0.5/max_mode_num)*(i)
#              # initialize randomly
#             elif omega_init == 2: 
#                  omega_init_= np.sort(np.exp(np.log(fs) + (np.log(0.5)-np.log(fs))*np.random.rand(max_mode_num,1))) 
#         else:
#             omega_init_ = np.zeros([max_iter_num])
#         return torch.from_numpy(omega_init_ )
        

    
