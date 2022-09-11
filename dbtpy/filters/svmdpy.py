# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:20:52 2020

@author: mozhenling
"""
import numpy as np

def svmd(f, alpha, tau, K, DC, init, tol, maxIterNum=100, omega_init=None):
    """
    u,u_hat,omega = svmd(f, alpha, tau, K, DC, init, tol, maxIterNum=10, omega_init=None)
    #--------------------------------------------------------------------------
    Python implementation by Zhenling Mo email: zhenlmo2-c@my.cityu.edu.hk
    The implementation is also based on the vmdpy of Vinícius Rezende Carvalho - vrcarva@gmail.com
    #--------------------------------------------------------------------------
    original paper:
        [1] M. Nazari and S. M. Sakhaei, “Successive variational mode decomposition,” Signal Processing,
            vol. 174, p. 107610, Sep. 2020, doi: 10.1016/j.sigpro.2020.107610.
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
    # ------------------ Part 1: Start initializing----------------------------
    #--------------- Mirroring the signal to extend
    # prevent zero denominator 
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
    
    # Maximum number of iterations (if not converged yet, then it won't anyway)
    Niter = 500
    # For future generalizations: individual alpha for each mode
    Alpha = alpha*np.ones(K)
    
   #-------- FFT of signal(and Hilbert transform concept=making it one-sided)
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
        
    
    
    # other inits
    uDiff = tol+np.spacing(1) # update step


    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = np.zeros([Niter, len(freqs), K],dtype=complex)
    
    # previously obtained modes
    sum_omega_converged_factor = 0
    sum_uk = 0 # accumulator
    n_max = np.zeros([K]) # store the convergence number of updates
    # convergence flag 
    # u_hat_plus_converged_flag = np.zeros([K])  # convergence flag for each mode
    #%% ------------------ Part 2: Main loop for iterative updates---------------
    for k in np.arange(K):
        # inner loop conter 
        n = 0 
        # update step
        udiff = tol+np.spacing(1)   
        # start with empty dual variables
        lambda_hat = np.zeros([Niter, len(freqs)], dtype = complex)
        #%%-------------- inner loop for mode extractions
        while(uDiff > tol and  n < Niter-1 ):
        #while(  n_inner < max_iter_num -1 ):    
            #----------- update u_hat_d
            u_hat_plus[n+1,:,k] = (f_hat_plus + (u_hat_plus[n,:,k] * \
            (Alpha[k]**2) * (freqs - omega_plus[n,k])**4) + lambda_hat[n,:] / 2) \
            /((1 + (Alpha[k]**2) * (freqs - omega_plus[n,k])**4) * \
            (1 + 2 * Alpha[k] * (freqs - omega_plus[n,k])**2 + \
            sum_omega_converged_factor))            
            #-----------update omega_d
            omega_plus[n+1,k] = np.dot(freqs[T//2:T],
                                                  (abs(u_hat_plus[n+1, T//2:T, k])**2))/np.sum(abs(u_hat_plus[n+1,T//2:T,k])**2)     
            #------ update lambda_hat
            lambda_hat[n+1,:] = lambda_hat[n,:] +  tau *  \
                ( f_hat_plus - ( u_hat_plus[n+1,:,k] + \
                ( (Alpha[k]**2 * (freqs -omega_plus[n+1,k])** 4) * \
                (f_hat_plus - u_hat_plus[n+1,:,k] - sum_uk + lambda_hat[n,:]/2)- \
                sum_uk / (1+2*Alpha[k]**2 * (freqs -omega_plus[n+1,k])**4)) + \
                sum_uk ) ) 
            # complete one update
            n = n + 1
            # converge yet?
            udiff = (1/T) * np.dot((u_hat_plus[n,:,k]-u_hat_plus[n-1,:,k]),np.conj((u_hat_plus[n,:,k]- \
            - u_hat_plus[n-1,:,k])))           
            udiff = abs(udiff) /(abs(np.dot(u_hat_plus[n-1,:,k], np.conj(u_hat_plus[n-1,:,k]))) + nan_inf_preventor)
            #omega_d = abs(omega_d[n_inner] - omega_d[n_inner - 1]) / abs(omega_d[n_inner])
        #%% 
        n_max[k] = n
        sum_uk = sum_uk+ u_hat_plus[n,:,k]
        sum_omega_converged_factor = sum_omega_converged_factor + 1 / ((Alpha[k]**2) * (freqs - omega_plus[n,k])**4)
        
    #%% ------------------ Part 2: Main loop for iterative updates---------------
    n_toll_max = int(max(n_max))
    # extend early converged mdoes to have the same length of last converged mode 
    for k in np.arange(K):
         omega_plus[int(n_max[k]):n_toll_max,k] = omega_plus[int(n_max[k]),k]
         u_hat_plus[int(n_max[k]):n_toll_max,:,k] = u_hat_plus[int(n_max[k]),:,k]
    
    #discard empty space if converged early
    Niter = np.min([Niter, n_toll_max])
    omega = omega_plus[: Niter,:]
    
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

      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    