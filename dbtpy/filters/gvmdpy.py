# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 21:15:22 2021

@author: mozhenling
"""

import numpy as np

def  gvmd(f, alpha, K, tau=0.0, DC=0, init=1, tol=1e-07, maxIterNum=5, omega_init=None):
    """
    u,u_hat,omega = gvmd(f, alpha, tau, K, DC, init, tol, maxIterNum=10, omega_init=None)
    #--------------------------------------------------------------------------
    Geralized Variational mode decomposition
    Python implementation by Zhenling Mo email: zhenlmo2-c@my.cityu.edu.hk
    The implementation is also based on the vmdpy of Vinícius Rezende Carvalho - vrcarva@gmail.com
    
    #--------------------------------------------------------------------------
    Input and Parameters:
    ---------------------
    f       - the time domain signal (1D) to be decomposed
    alpha   - the balancing parameter of the data-fidelity constraint
    tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    K       - the number of modes to be recovered
    DC      - true if the first mode is put and kept at DC (0-freq)
    init    - 0 = all omegas start at 0
              1 = all omegas start uniformly distributed
              2 = all omegas initialized randomly
    tol     - tolerance of convergence criterion; typically around 1e-6
    maxIterNum    - the maximum number of iteration
    omega_init    - if it is not none, initialize by it. Otherwise, initialize by init. 

    Output:
    -------
    u       - the collection of decomposed modes
    u_hat   - spectra of the modes
    omega   - estimated mode center-frequencies
    #--------------------------------------------------------------------------
    #------ you may set some parameters as follows    
    tau = 0.           # noise-tolerance (no strict fidelity enforcement)
    K = 3              # 3 modes
    DC = 0             # no DC part imposed
    init = 3           # initialize omegas uniformly
    tol = 1e-7
    maxIterNum = 10   #maximun iteration number
    #--------------------------------------------------------------------------
  """
  
    #-------------------------------------------
    # Maximum number of iterations (if not converged yet, then it won't anyway)
    Niter =  maxIterNum
    
    nan_inf_preventor = np.spacing(1)
    if len(f)%2:
       f = f[:-1]

    # Period and sampling frequency of input signal
    fs = 1./len(f)
    
    ltemp = len(f)//2 
    fMirr =  np.append(np.flip(f[:ltemp],axis = 0),f)  
    fMirr = np.append(fMirr,np.flip(f[-ltemp:],axis = 0))

    # Time Domain 0 to T (of mirrored signal)
    T = len(fMirr)
    t = np.arange(1,T+1)/T  
    
    # Spectral Domain discretization
    freqs = t-0.5-(1/T)

   
    # For future generalizations: individual alpha for each mode
    Alpha = alpha*np.ones(K)
    
    # Construct and center f_hat
    f_hat = np.fft.fftshift((np.fft.fft(fMirr)))
    f_hat_plus = np.copy(f_hat) #copy f_hat
    f_hat_plus[:T//2] = 0

    # Initialization of omega_k
    omega_plus = np.zeros([Niter, K])


    if omega_init is not None :
        # use the most recent updates
        omega_plus[0,:] = omega_init
    elif init == 1:
        for i in range(K):
            omega_plus[0,i] = (0.5/K)*(i)
    elif init == 2:
        omega_plus[0,:] = np.sort(np.exp(np.log(fs) + (np.log(0.5)-np.log(fs))*np.random.rand(1,K)))
    else:
        omega_plus[0,:] = 0
            
    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0,0] = 0
    
    # start with empty dual variables
    lambda_hat = np.zeros([Niter, len(freqs)], dtype = complex)
    
    # other inits
    uDiff = tol+np.spacing(1) # update step
    n = 0 # loop counter
    sum_uk = 0 # accumulator
    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = np.zeros([Niter, len(freqs), K],dtype=complex)   

    #*** Main loop for iterative updates***
    # counting the number of converged modes
    mode_out_flag = 0 
    # previously obtained modes
    sum_omega_converged_factor = 0
    # convergence flag 
    u_hat_plus_converged_flag = np.zeros([K])  # convergence flag for each mode
    # toll_fre , frequency resolution convergence creterion
    tol_fre = 1./T
    # if mode_out_flag = K, or n = Niter -1 , stop updating
    while ( n < Niter-1 ): # not converged and below iterations limit
    
        for k in np.arange(K):
            #%% updata first mode (also the first mode to be out)
            if k==0: 
               # the first mode is supposed to be out first, no mode converged 
               # u_hat_plus[n,:,K-1] has been updated if n > 0
               sum_uk = u_hat_plus[n,:,K-1] + sum_uk - u_hat_plus[n,:,k] 
               # if converged, just copy the mode to update
               if u_hat_plus_converged_flag[k] == 1:
                   u_hat_plus[n+1,:,k] = u_hat_plus[n,:,k]
                   omega_plus[n+1,k] = omega_plus[n,k]
               else:
                   # update spectrum of first mode through Wiener filter of residuals
                   u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk #- sum_uk_converged 
                                          + lambda_hat[n,:]/2)/(1.+Alpha[k]*(freqs - omega_plus[n,k])**2 + sum_omega_converged_factor)
                   # update first omega if not held at 0
                   if not(DC):
                       omega_plus[n+1,k] = np.dot(freqs[T//2:T],
                                                  (abs(u_hat_plus[n+1, T//2:T, k])**2))/np.sum(abs(u_hat_plus[n+1,T//2:T,k])**2)
               
               
            #%%----------------- update of any other mode--------------------------
            else: # k > 0
                #accumulator
                sum_uk = u_hat_plus[n+1,:,k-1] + sum_uk - u_hat_plus[n,:,k]
                # mode spectrum
                # if converged, just copy the mode to update
                if u_hat_plus_converged_flag[k] == 1:
                   u_hat_plus[n+1,:,k] = u_hat_plus[n,:,k]
                   omega_plus[n+1,k] = omega_plus[n,k]
                else:
                    u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk #- sum_uk_converged 
                                           + lambda_hat[n,:]/2)/(1+Alpha[k]*(freqs - omega_plus[n,k])**2 + sum_omega_converged_factor)
                    # center frequencies
                    omega_plus[n+1,k] = np.dot(freqs[T//2:T],
                                               (abs(u_hat_plus[n+1, T//2:T, k])**2))/np.sum(abs(u_hat_plus[n+1,T//2:T,k])**2)
             
        #%% Dual ascent
        lambda_hat[n+1,:] = lambda_hat[n,:] + tau*(f_hat_plus - np.sum(u_hat_plus[n+1,:,:],axis = 1)) #- sum_uk_converged)
        
        #%% converged yet ?
        # normalized convergence
        for k in range(K):
            if u_hat_plus_converged_flag[k] == 0:
                uDiff_fre = np.abs(omega_plus[n+1,k] - omega_plus[n,k])
                uDiff = np.abs(np.dot((u_hat_plus[n+1,:,k]-u_hat_plus[n,:,k]),
                                np.conj((u_hat_plus[n+1,:,k]-u_hat_plus[n,:,k]))))# / (np.abs(np.dot(u_hat_plus[n,:,k],np.conj(u_hat_plus[n,:,k])))+ nan_inf_preventor)
                if  uDiff < tol and uDiff_fre < tol_fre: 
                    # set converged flag
                    u_hat_plus_converged_flag[k] = 1
                    # save the converged central frequency info. to the denomenator
                    # sum_omega_converged_factor = sum_omega_converged_factor + 1.0 / ((Alpha[k]**2)*(freqs - omega_plus[n+1,k])**4)
                    # conunt converged modes
                    mode_out_flag = mode_out_flag + 1
    
        #%% loop counter
        n = n+1
        # if converged ealy
        if mode_out_flag == K:
            break
    #%% Postprocessing and cleanup
    
    #discard empty space if converged early
    Niter = np.min([Niter,n])
    omega = omega_plus[:Niter,:]
    
    idxs = np.flip(np.arange(1,T//2+1),axis = 0)
    # Signal reconstruction
    u_hat = np.zeros([T, K],dtype = complex)
    u_hat[T//2:T,:] = u_hat_plus[Niter-1,T//2:T,:]
    u_hat[idxs,:] = np.conj(u_hat_plus[Niter-1,T//2:T,:])
    u_hat[0,:] = np.conj(u_hat[-1,:])    
    
    u = np.zeros([K,len(t)])
    # u = np.real(np.fft.ifft(np.fft.ifftshift(u_hat,axis= 0), axis=0))
    for k in range(K):
        u[k,:] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,k])))
        
    # remove mirror part
    u = u[:,T//4:3*T//4]

    # recompute spectrum
    u_hat = np.zeros([u.shape[1],K],dtype = complex)
    # u_hat = np.fft.fftshift(np.fft.fft(u,axis= 0),axis= 0)
    for k in range(K):
        u_hat[:,k]=np.fft.fftshift(np.fft.fft(u[k,:]))
        
    # For convenience here: Order omegas increasingly and reindex u/u_hat
    sortIndex = np.argsort(omega[-1,:])
    omega = omega[:,sortIndex]
    u_hat = u_hat[:,sortIndex]
    u = u[sortIndex,:]

    return u, u_hat, omega
