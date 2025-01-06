import numpy as np # math pack!
from scipy import signal as sp_signal, interpolate as sp_interpolate
from dbtpy.tools import visual_tools as sig_plots
# import matplotlib.pyplot as plt # plotting
import matplotlib
matplotlib.use('Qt5Agg')


def tacho_to_rpm(tacho, fs, PPR=1, threshold=None, slope=2):
    """
    Convert tachometer impulse series to revolution-per-minute series

    Inputs:
        tacho: tachometer impulse series
        fs: sampling frequency
        PPR: pulses per revolution, resolution of the tachometer
        slope: 2 or -2, 2 means using positive slope and -2 means using negative slope to indicate the change of impulses
    Outputs:
       rpm: revolution-per-minute time series (unit: rpm), meaning the amount of revolutions in a minute

    Ref.: https://github.com/efsierraa/PyCycloVarREB/blob/master/functions/REB_functions.py
    """

    # time resolution is the inverse of frequency resolution
    dt = 1/fs
    # get time axis stamps (unit: second)
    t = np.arange(0, len(tacho)) * dt
    # Produce +1 where signal is above trigger level
    # and -1 where signal is below trigger level
    trigger_level = threshold if threshold is not None else np.mean(tacho)
    xs = np.sign(tacho - trigger_level)
    # Differentiate this to find where xs changes
    # between -1 and +1 and vice versa
    xDiff = np.diff(xs) # a sequence of 2,0,or -2; 2 means a positive change, -2 means a negative change

    # We need to synchronize xDiff with variable t from the
    # code above, since DIFF shifts one step
    tDiff=t[1:]

    # Now find the time instances of positive slope positions
    # (-2 if negative slope is used)
    tTacho = tDiff[xDiff == slope]  # xDiff.T return the indexes boolean
    # Count the time between the tacho signals and compute
    # the RPM at these instances
    # rev_per_impulse = 1/PPR # how many revolutions in one impulse
    # impulse_per_sec = 1/np.diff(tTacho) # how many impulses in one second
    # sec_per_min = 60 # how many seconds in one minute
    # rpmt_ = sec_per_min * rev_per_impulse * impulse_per_sec # how many revolutions in one minute

    rpm = 60 / PPR / np.diff(tTacho)  # Temporary rpm values

    if len(rpm) <3:
        raise ValueError('The length of tachometer impulses is too short for estimation and interpolation')
    else:
        # Use three tacho pulses at the time and assign mean
        # value to the center tacho pulse
        rpm = 0.5 * (rpm[0:-1] + rpm[1:])
        tTacho = tTacho[1:-1]  # diff again shifts one sample
        # Smoothing
        wfiltsv = int(2 ** np.fix(np.log2(.05 * fs))) - 1
        if len(rpm)>wfiltsv:
            rpm = sp_signal.savgol_filter(rpm, wfiltsv, 2)  # smoothing filter
        # instantiate an interpolator
        # Fits a spline y = spl(x) of degree k to the provided x, y data. k=1, piece wise linear
        # Spline function passes through all provided points. Equivalent to UnivariateSpline with s = 0.
        rpmt_interpolator = sp_interpolate.InterpolatedUnivariateSpline(x=tTacho, y=rpm, w=None, bbox=[None, None], k=1)
        # Evaluate the fitted spline at the given points
        rpm = rpmt_interpolator(t)

    return rpm, t


def angular_resampling(t, rpm, sig_t, keepLen=False,reLen=None):
    """
    Computed order tracking / Angular resampling
    
    Inputs:
        t: time stamps (unit: second)
        rpm: revolution-per-minute time series (unit: rpm), meaning the amount of revolutions in a minute
        sig_t: signal in time domain
        keepLen: sig_cyc have the same length of sig_t

    outputs:
        sig_cyc: signal in cycle domain, which is angular-resampled from sig_t
        fs_cyc: sampling frequency in cycle domain. If none, use the resolution fs_cyc = int(1 / (dt * min(rpm_in_hz)))
                fs_cyc>=2*order_highest:
                      fs_cyc should be at least twice larger than the highest order in order spectrum (Nyquist sampling frequency).
                      fs_cyc>=reSample_num*order_resolution:
                      fs_cyc should ensure enough order_resolution to differentiate orders in order spectrum.
                      reSample_num = int(fs_cyc*max(cumulative_phase))
                      order_resolution = int(1/max(cumulative_phase))

    Ref.:
    [1] https://doi.org/10.3390/s24020454
    [2] W. Cheng, R. X. Gao, J. Wang, T. Wang, W. Wen, and J. Li,
        “Envelope deformation in computed order tracking and error in order
        analysis,” Mechanical Systems and Signal Processing, vol. 48, no. 1–2,
        pp. 92–102, Oct. 2014, doi: 10.1016/j.ymssp.2014.03.004.
    """
    # The inputs should correspond to the same sampling frequency in time domain.
    # Thus, they should have the same lengths. Otherwise, raise the error and the message.
    assert len(t) ==len(rpm) ==len(sig_t), "The inputs should correspond to the same sampling frequency in time domain!"

    rpm_in_hz = rpm / 60 # convert rpm unit to revolution-per-second or equivalently Hz

    # Time resolution
    dt = t[1] - t[0] # 1/fs

    # Calculate cumulative phase of the shaft， integral rpm over time
    cumulative_phase = np.cumsum(rpm_in_hz * dt)
    cumulative_phase -= cumulative_phase[0] # zero phase at the starting point

    # Determine angular sampling frequency and points
    if reLen is None and not keepLen:
        if min(rpm_in_hz) == 0 or min(rpm_in_hz) < 0:
            raise ValueError('The rpm of the reference should not be 0 or negative!')
        fs_cyc = int(1 / (dt * min(rpm_in_hz)))
        reSampleNum= int(fs_cyc * max(cumulative_phase))
    else:
        reSampleNum = len(sig_t) if keepLen else reLen
        fs_cyc = int(reSampleNum / max(cumulative_phase))  # points per phase

    # Generate constant phase intervals
    constant_phase_intervals = np.linspace(start=0, stop=max(cumulative_phase), num=reSampleNum)

    #-- Angular resampling process: fit samples using interpolation, and re-select samples from fitted functions
    # Interpolate to find new time points with constant phase intervals
    interp_func = sp_interpolate.interp1d(cumulative_phase, t, kind='linear') # fit cumulative_phase and t
    # interp_func = sp_interpolate.UnivariateSpline(cumulative_phase, t, k=3, s=0)  # fit cumulative_phase and t
    times_of_constant_phase_intervals = interp_func(constant_phase_intervals) # re-select t based on constant_phase_intervals
    # Use UnivariateSpline for spline interpolation
    # Fits a spline y = spl(x) of degree k to the provided x, y data. k=3 by default, a cubic spline.
    # If s=0, spline will interpolate through all data points. This is equivalent to InterpolatedUnivariateSpline.
    spline_interpolator = sp_interpolate.UnivariateSpline(x=t, y=sig_t, k=3, s=0) # fit t and sig_t
    # Evaluate the fitted spline at the given points
    sig_cyc = spline_interpolator(times_of_constant_phase_intervals) # re-select sig_t based on times_of_constant_phase_intervals

    return sig_cyc, fs_cyc



class Example():
    def __init__(self):
        # sampling frequency of the time domain signals
        self.fs = 512  # Hz
        # duration in second of time domain signals
        self.duration = 1  # second
        self.num_points = int(self.fs * self.duration)
        # time resolution
        self.dt = 1 / self.fs
        # pulses per revolution, resolution of the tachometer encoder
        self.PPR = 8
        # shaft speed frequency, order = 1 as the reference is itself.
        self.f_shaft = 1  # 1 Hz = 60 rpm
        # speed scaling factor
        self.start_factor = 1
        self.stop_factor = 3
        # other orders considered
        self.order_list = [1, 5, 10, 15, 20, 25, 30, 35]

        self.rpm, self.tacho, self.sig_t, self.sig_cyc, self.fs_cyc = None, None, None, None, None

    def angular_resampling(self, img_show=True, save_root = './ang', same_amps=True, rand_samps =False, varying_func=None,
                                  use_tacho=True, fontsize = 10, format = 'png', figsize = (4, 3), dpi = 300):
        """
        Simulate some signals to demonstrate the angular resampling
        Inputs:
            img_show: plot and show images
            img_save: save the plotted images
            img_path: directory to save the images
            same_amps: all orders have the same sin amplitudes if true
            rand_samps: draw random variables for varying factors if true
            use_tacho: simulate tachometer signal for speed estimation if true; otherwise, use the shaft speed directly
            others: settings for formatting plots

        Outputs:
            (tacho, sig_t, fs, dt, df): time domain signals, their sampling frequency, time and frequency resolutions
            (sig_cyc, fs_cyc, dcyc, dorder): cycle domain signal, its sampling frequency, cycle and order resolutions
        """
        #------------------------------------------------------------------------
        #-------------- customize the simulated signal
        # sampling frequency of the time domain signals
        fs = self.fs  # Hz
        # duration in second of time domain signals
        duration = self.duration  # second
        num_points = self.num_points
        # time resolution
        dt = self.dt
        # pulses per revolution, resolution of the tachometer encoder
        PPR = self.PPR
        # shaft speed frequency, order = 1 as the reference is itself.
        f_shaft = self.f_shaft # 1 Hz = 60 rpm
        # speed scaling factor
        start_factor = self.start_factor
        stop_factor = self.stop_factor
        # other orders considered
        order_list = self.order_list
        order_amps = [1 for _ in order_list] if same_amps else [2*(i+1) for i in range(len(order_list))]
        # speed factor varying profile
        def varying_func_in(v):
            # High frequency components are hard to be estimated accurately in the interpolations of angular resampling.
            # Errors in resampled time and amplitudes of high frequency components will lead to high noise floors in the
            # order spectrum. In addition, make sure we have an enough sampling frequency/resolution to analyze higher
            # frequency/order components.
            return v+v**2
        # generate time domain signals
        def time_domain_signals():
            # factor inputs
            if rand_samps:
                rv = np.random.uniform(low=start_factor, high=stop_factor, size=num_points)
                v = np.sort(rv)
            else:
                v = np.linspace(start_factor, stop_factor, num_points)
            varying_speed_factors = varying_func_in(v) if varying_func is None else varying_func(v)
            # Creating phases
            shaft_phase = 2 * np.pi * np.cumsum(varying_speed_factors * f_shaft * dt)
            # Simulate tachometer signal to estimate speed in rpm
            if use_tacho:
                encoder_phase = PPR * 2 * np.pi * np.cumsum(varying_speed_factors * f_shaft * dt)
                tacho = np.sin(encoder_phase)
                rpm, t_ = tacho_to_rpm(tacho, fs, PPR=PPR)
            else:
                rpm = varying_speed_factors * f_shaft * 60 # times 60 to get revolution-per-minute
                tacho = None

            sig_t = 0
            for order, amp in zip(order_list, order_amps):
                # original signal
                sig_t += amp*np.sin(order * shaft_phase)
            # nonsync_phase = 2 * np.pi * f2 * t
            # sig_t += np.sin(nonsync_phase)
            # # reverse speed changing
            # tacho = tacho[::-1]
            # sig_t = sig_t[::-1]
            # get rpm inputs
            return rpm, tacho, sig_t, shaft_phase

        # ------------------------------------------------------------------------
        #-------------- subsequent processing
        # time stamps
        t = np.linspace(start=0, stop=duration, num=num_points)
        # get time domain signals
        rpm, tacho, sig_t, shaft_phase = time_domain_signals()
        # normalization
        sig_t /=max(np.abs(sig_t))
        # get resampled signal
        sig_cyc, fs_cyc = angular_resampling(t, rpm, sig_t, keepLen=True)
        # normalization
        sig_cyc /=max(np.abs(sig_cyc))
    
        f, sig_f_amp = sig_plots.fftspec(fs, sig_t)  # signal in the frequency domain
        order, sig_order_amp = sig_plots.fftspec(fs_cyc, sig_cyc)
    
        cyc = np.linspace(0, len(sig_cyc)/fs_cyc, len(sig_cyc))  # cycle vector
    
    
        #-- plot and save
        if img_show and save_root is not None:
            sig_plots.formated_plot(x=t, y=rpm, xlabel='Time (s)', ylabel='Speed (rpm)',
                                save_dir=save_root, filename= 'xydata_1_Speed', figsize=figsize, fontsize=fontsize,
                                dpi=dpi, format=format, title='Varying speed')
    
            sig_plots.formated_plot(x=t, y=shaft_phase, xlabel='Time (s)', ylabel='Phase (radian)',
                                save_dir=save_root, filename= 'xydata_2_phase', figsize=figsize, fontsize=fontsize,
                                dpi=dpi, format=format, title='Varying shaft phase')
    
            if use_tacho:
                sig_plots.formated_plot(x=t, y=tacho, xlabel='Time (s)', ylabel='Normalized Amplitude',
                                    save_dir=save_root, filename= 'xydata_3_Tachormeter',figsize=figsize,fontsize=fontsize,
                                    dpi=dpi,format=format, title='Tachometer signal')
    
            sig_plots.formated_plot(x=t, y=sig_t, xlabel='Time (s)', ylabel='Normalized Amplitude',
                                save_dir=save_root, filename= 'xydata_4_Time', figsize=figsize, fontsize=fontsize,
                                dpi=dpi, format=format, title='Time Domain')
    
            sig_plots.formated_plot(x=f, y=sig_f_amp, xlabel='Frequency (Hz)', ylabel='Normalized Amplitude',
                                save_dir=save_root, filename= 'xydata_5_Freq', figsize=figsize, fontsize=fontsize,
                                dpi=dpi, format=format, title='Frequency Domain')
    
            sig_plots.formated_plot(x=cyc, y=sig_cyc, xlabel='Cycle', ylabel='Normalized Amplitude',
                                save_dir=save_root, filename= 'xydata_6_Cycle', figsize=figsize, fontsize=fontsize,
                                dpi=dpi, format=format, title='Cycle Domain')
    
            sig_plots.formated_plot(x=order, y=sig_order_amp, xlabel='Order', ylabel='Normalized Amplitude',
                                save_dir=save_root, filename= 'xydata_7_Order', figsize=figsize, fontsize=fontsize,
                                dpi=dpi, format=format, title='Order Dommain')
        (self.rpm, self.tacho, self.sig_t, self.fs), (self.sig_cyc, self.fs_cyc) = (rpm, tacho, sig_t, fs), (sig_cyc, fs_cyc)
        return (rpm, tacho, sig_t, fs), (sig_cyc, fs_cyc)


if __name__ == "__main__":
    # #--- ang-same_amps-no_tacho-rand_equal_samps
    # figsize=(4,3)
    # same_amps = True   # same_amps: different amplitudes for different orders at initialization
    # use_tacho = False  # no_tacho: use shaft speed directly
    # rand_samps = False # equal_samps: use equally spaced samples of scale-varying factors
    # # show and save
    # img_show = True
    # save_root = r'same_amps-no_tacho-equal_samps'
    # # seed=1
    # # np.random.seed(seed)
    # eg1 = Example()
    # (rpm, tacho, sig_t, fs), (sig_cyc, fs_cyc) = \
    #     eg1.angular_resampling(img_show =img_show, save_root=save_root,figsize=figsize,use_tacho=use_tacho,
    #                                same_amps=same_amps, rand_samps=rand_samps)

    # --- ang-diff_amps-no_tacho-rand_samps
    figsize = (4, 3)
    same_amps = False  # diff_amps: different amplitudes for different orders at initialization
    use_tacho = False  # no_tacho: use shaft speed directly
    rand_samps = False # equal_samps: use equally spaced samples of scale-varying factors
    # show and save
    img_show = True
    save_root = r'diff_amps-no_tacho-equal_samps'
    # seed=1
    # np.random.seed(seed)
    eg2 = Example()
    (rpm_, tacho_, sig_t_, fs_), (sig_cyc_, fs_cyc_) = \
        eg2.angular_resampling(img_show=img_show, save_root=save_root, figsize=figsize, use_tacho=use_tacho,
                                   same_amps=same_amps, rand_samps=rand_samps)




