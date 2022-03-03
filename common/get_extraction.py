import numpy as np
from scipy import signal
import math

def nexpow2(x):
    """
    :param x:
    :return:
    """
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

def rolling_window(data, window_len, noverlap=1, axis=-1, padded=False, copy=True):
    """
    Calculate a rolling window over a data
    :param data: numpy array. The array to be slided over.
    :param window_len: int. The rolling window size
    :param noverlap: int. The rolling window stepsize. Defaults to 1.
    :param padded:
    :param axis: int. The axis to roll over. Defaults to the last axis.
    :param copy: bool. Return strided array as copy to avoid sideffects when manipulating the
        output array.
    :return:
    numpy array
        A matrix where row in last dimension consists of one instance
        of the rolling window.
    See Also
    --------
    pieces : Calculate number of pieces available by rolling
    """

    data = data if isinstance(data, np.ndarray) else np.array(data)
    assert axis < data.ndim, "Array dimension is less than axis"
    assert noverlap >= 1, "Minimum overlap cannot be less than 1"
    assert window_len <= data.shape[axis], "Window size cannot exceed the axis length"
    
    arr_shape = list(data.shape)
    arr_shape[axis] = np.floor(data.shape[axis] / noverlap - window_len / noverlap + 1).astype(int)
    arr_shape.append(window_len)

    strides = list(data.strides)
    strides[axis] *= noverlap
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=arr_shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided

def gain_control(gain, constraint_len, window = 'hann', axis = -1):
    """
    This function gets the gain and applies a constraint on the impulse repsonse
    to satisfy the linear convolution property. This will overcome the time-aliasing
    problem.
    Ref: P. Scalart (2008)
    :param gain:
    :param constraint_len:
    :param window:
    :param axis:
    :return:
    """

    gain_mean = np.mean(gain, axis = axis)
    nfft = gain.shape[axis]
    l2 = np.fix(constraint_len/2).astype(int)
    window = signal.get_window(window, constraint_len)
    gain_impulse = np.real(np.fft.ifft(gain))
    p0 = gain_impulse[:l2] * window[l2:]
    p1 = np.zeros([nfft-constraint_len])
    p2 = gain_impulse[nfft-l2:] * window[:l2]
    new_gain_impulse = np.hstack((p0, p1, p2))    # % Time -> Frequency
    new_gain = abs(np.fft.fft(new_gain_impulse,nfft))
    mean_new_gain = np.mean(new_gain, axis = axis)
    new_gain = new_gain * np.sqrt(gain_mean/mean_new_gain) 
    return new_gain

def moving_average(array, w : int = 5, axis = -1):
    """
    Moving average smoothing of an array for smoothing the noisy fluctuations
    :param array: numpy array. The array to be smoothed over.
    :param w: int. number of points for averaging window, constant, default = 5
    :param axis:
    :return:
    smooth data array (the number of w first points are zero, averaging starts after the "w" first points)
    """

    mov_mean = np.cumsum(array, axis = axis)
    mov_mean[:, w:] = mov_mean[:, w:] - mov_mean[:, :-w]
    return mov_mean/w

def wiener_filter(strain, noise, fs = 4096, nfft = None, nperseg = None, noverlap = None, 
             window = 'hann', filt_type = 'hrnr', smooth = True, axis = -1):
    """

    :param strain:
    :param noise:
    :param fs:
    :param nfft:
    :param nperseg:
    :param noverlap:
    :param window:
    :param filt_type:
    :param smooth:
    :param axis:
    :return:
    """
    if strain.shape[0] > strain.shape[-1]:
        strain = strain.T
        noise = noise.T
    num_detector, strain_length = strain.shape
    noise_length = noise.shape[axis]
    
    if not nperseg:
        nperseg = np.fix(0.02*fs).astype(int)
    if np.remainder(nperseg, 2) == 1:
        nperseg = nperseg + 1
   
    if not nfft:
        nfft = max([256, nexpow2(nperseg)])
    
    if not noverlap:
        shift = 0.3
        norm_coeff = 1/shift
        noverlap = np.fix((1-shift) * nperseg).astype(int)
        offset = nperseg - noverlap
        # n_times = np.fix((strain_length - nfft)/offset).astype(int)
    else:
        n_times = np.fix((strain_length - nfft)/offset).astype(int)
        offset = nperseg - noverlap            

    window = signal.get_window(window, nperseg)

    # smoothing factor for recursive averaging between priori and posteriori frames
    alpha_noise = 0.95
    beta_noise = 0.20
    gamma_noise = 0.85
    lambda_main = 0.95
    
    noise_slice = rolling_window(noise, window_len = nperseg, noverlap=offset)
    init_noise = np.zeros([num_detector, nfft])
    for count in range(noise_slice.shape[1]):
        init_noise += abs(np.fft.fft(noise_slice[:, count, :]*window, nfft, axis = axis))
    init_noise = init_noise/len(init_noise)
    tsnr_output = np.zeros([num_detector, strain_length])
    strain_slice = rolling_window(strain, window_len = nperseg, noverlap = offset, padded=True)
    n_times = strain_slice.shape[1]
    strain_amplitude = np.zeros([num_detector, nfft, n_times])
    strain_phase = np.zeros([num_detector, nfft, n_times])
    noise_psd = np.zeros([num_detector, nfft, n_times])
    tsnr_amplitude = np.zeros([num_detector, nfft, n_times])
    tsnr_copy = np.zeros([num_detector, nfft, n_times], dtype='complex')
    hrnr_prio = np.zeros([num_detector, nfft, n_times])
    for count in range(strain_slice.shape[1]):
        fft_slice = np.fft.fft(strain_slice[:, count, :]*window, nfft, axis = axis)
        strain_amplitude[:, :, count] = abs(fft_slice)
        strain_phase[:, :, count] = np.angle(fft_slice)
        ## TODO: @Amin add the noise estimation on a network of arrays and update the algorithm
        if count == 0:
            noise_psd[:, :, count] = init_noise
            priori_snr = np.zeros_like(init_noise)
            priori_noise = init_noise
        else:
            post_noise_temp = strain_amplitude[:, :, count]
            priori_noise = noise_psd[:, :, count-1]
            recursive_noise = alpha_noise * priori_noise + (1 - alpha_noise) * post_noise_temp
            noise_mask = np.argwhere((recursive_noise * beta_noise < priori_noise) |
                               (recursive_noise * gamma_noise > priori_noise))
            noise_psd[:, :, count] = recursive_noise
            noise_psd[noise_mask[:, 0], noise_mask[:, 1], count] = noise_psd[noise_mask[:, 0], noise_mask[:, 1], count-1]
        
        # posteriori_snr
        post_snr = strain_amplitude[:, :, count] / noise_psd[:, :, count]   
        # limitation to prevent distortion
        post_snr = np.where(post_snr <= 0.1, 0.1, post_snr)
        # estimating priori_snr using decision-directed method
        # recursive 1st order average method for gain estimation based on
        # decision-directed and wiener gain function
        eta_gain = lambda_main * (priori_snr / priori_noise) + (1 - lambda_main) * post_snr  
        new_amplitude = (eta_gain / (eta_gain + 1)) * strain_amplitude[:, :, count]
        
        """
        two-step noise reduction (TSNR) algorithm
        Ref: Plapous et al (2004, May), A two-step noise reduction technique. 
        DOI: 10.1109/ICASSP.2004.1325979
        """

        tsnr_post = new_amplitude / noise_psd[:, :, count]
        gain_tsnr = tsnr_post / (tsnr_post + 1)  
        # wiener gain function for estimation of tSNR 
        tsnr_amplitude[:, :, count] = gain_tsnr
        for idx in range(num_detector):
            gain_tsnr[idx, :] = gain_control(gain_tsnr[idx, :], constraint_len = (nfft//2))
        
        new_amplitude = gain_tsnr * new_amplitude
        # update priori_snr for the next frame
        priori_snr = new_amplitude
        # copy for hrnr estimation
        if filt_type == 'hrnr':
            hrnr_prio[:, :, count] = new_amplitude
        
        tsnr_copy[:, :, count] = new_amplitude * np.exp(1j*strain_phase[:, :, count])
        temp_out = np.real(np.fft.ifft(tsnr_copy[:, :, count], nfft, axis = axis))/norm_coeff
        try:
            tsnr_output[:, count*offset:count*offset+nfft] = tsnr_output[:, count*offset:count*offset+nfft] + temp_out
        except:
            continue
    
    tsnr_slice = rolling_window(tsnr_output, window_len = nperseg, noverlap=offset)
    hrnr_output = np.zeros([num_detector, strain_length])
    hrnr_amplitude = np.zeros([num_detector, nfft, n_times])
    hrnr_copy = np.zeros([num_detector, nfft, n_times], dtype='complex')
    for count in range(tsnr_slice.shape[1]): 
        fft_slice = np.fft.fft(tsnr_slice[:, count, :]*window, nfft, axis = axis)
        hrnr_amplitude[:, :, count] = abs(fft_slice)
        # estimating priori_snr using decision-directed method
        # recursive 1st order average method for gain estimation based on
        # decision-directed and wiener gain function
        eta_hrnr = tsnr_amplitude[:, :, count] * hrnr_prio[:, :, count] + ((1 - tsnr_amplitude[:, :, count]) * hrnr_amplitude[:, :, count]) / noise_psd[:, : , count]
        gain_hrnr = eta_hrnr/(eta_hrnr+1)
        hrnr_amplitude_out = gain_hrnr * strain_amplitude[:, :, count]
        hrnr_copy[:, :, count] = hrnr_amplitude_out * np.exp(1j*strain_phase[:, :, count])
        temp_out = np.real(np.fft.ifft(hrnr_copy[:, :, count], nfft, axis = axis))/norm_coeff
        try: 
            hrnr_output[:, count*offset:count*offset+nfft] = hrnr_output[:, count*offset:count*offset+nfft] + temp_out
        except:
            continue
    if smooth:
        tsnr_output = moving_average(tsnr_output)
        hrnr_output = moving_average(hrnr_output)    
    if filt_type == 'hrnr':
        return hrnr_output
    elif filt_type == 'tsnr':
        return tsnr_output
