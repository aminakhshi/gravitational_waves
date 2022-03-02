import numpy as np
import math
from scipy import signal

def get_slice(strain, merger_time, window_size = None, step = 1, axis = -1):
    if not window_size:
        fs = 4096
        window_size = 2 * fs
    # for ifo, val in enumerate(strain):
    bound_low = np.floor(merger_time - np.floor(window_size/2)).astype(int)
    bound_up = np.floor(merger_time + np.floor(window_size/2)).astype(int)
    return strain.take(indices=range(bound_low, bound_up, step), axis=axis)

def get_slice_noise(strain, merger_time, cut_window = None, step = 1, axis = -1):
    if not cut_window:
        fs = 4096
        cut_window = [9*fs, 1*fs]
    # for ifo, val in enumerate(strain):
    bound_low = np.floor(merger_time - cut_window[0]).astype(int)
    bound_up = np.floor(merger_time - cut_window[-1]).astype(int)
    return strain.take(indices=range(bound_low, bound_up, step), axis=axis)
# strain[:, bound_low:bound_up]

def nexpow2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

def rolling_window(data, window_len, noverlap=1, padded=False, axis=-1, copy=True):
    """
    Calculate a rolling window over a data
    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    window_len : int
        The rolling window size
    noverlap : int
        The rolling window stepsize. Defaults to 1.
    axis : int
        The axis to roll over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the rolling window.
    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.
    Examples
    --------
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> rolling_window(a, window_len=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> rolling_window(a, window_len=3, noverlap=2)
    array([[1, 2, 3],
           [3, 4, 5]])
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
    
# =============================================================================
# Filters taken from pycbc
# Reference: https://github.com/gwastro/pycbc
# =============================================================================
def fir_zero_filter(coeff, timeseries):
    """Filter the timeseries with a set of FIR coefficients

    Parameters
    ----------
    coeff: numpy.ndarray
        FIR coefficients. Should be and odd length and symmetric.
    timeseries: pycbc.types.TimeSeries
        Time series to be filtered.

    Returns
    -------
    filtered_series: pycbc.types.TimeSeries
        Return the filtered timeseries, which has been properly shifted to account
    for the FIR filter delay and the corrupted regions zeroed out.
    """
    # apply the filter
    series = signal.lfilter(coeff, 1.0, timeseries)

    # reverse the time shift caused by the filter,
    # corruption regions contain zeros
    # If the number of filter coefficients is odd, the central point *should*
    # be included in the output so we only zero out a region of len(coeff) - 1
    series[:(len(coeff) // 2) * 2] = 0
    series.roll(-len(coeff)//2)
    return series

def notch_fir(timeseries, f1, f2, order, beta=5.0):
    """ notch filter the time series using an FIR filtered generated from
    the ideal response passed through a time-domain kaiser window (beta = 5.0)

    The suppression of the notch filter is related to the bandwidth and
    the number of samples in the filter length. For a few Hz bandwidth,
    a length corresponding to a few seconds is typically
    required to create significant suppression in the notched band.
    To achieve frequency resolution df at sampling frequency fs,
    order should be at least fs/df.

    Parameters
    ----------
    Time Series: TimeSeries
        The time series to be notched.
    f1: float
        The start of the frequency suppression.
    f2: float
        The end of the frequency suppression.
    order: int
        Number of corrupted samples on each side of the time series
        (Extent of the filter on either side of zero)
    beta: float
        Beta parameter of the kaiser window that sets the side lobe attenuation.
    """
    k1 = f1 / float((int(1.0 / timeseries.delta_t) / 2))
    k2 = f2 / float((int(1.0 / timeseries.delta_t) / 2))
    coeff = signal.firwin(order * 2 + 1, [k1, k2], window=('kaiser', beta))
    return fir_zero_filter(coeff, timeseries)


def lowpass_fir(timeseries, frequency, order, beta=5.0):
    """ Lowpass filter the time series using an FIR filtered generated from
    the ideal response passed through a kaiser window (beta = 5.0)

    Parameters
    ----------
    Time Series: TimeSeries
        The time series to be low-passed.
    frequency: float
        The frequency below which is suppressed.
    order: int
        Number of corrupted samples on each side of the time series
    beta: float
        Beta parameter of the kaiser window that sets the side lobe attenuation.
    """
    k = frequency / float((int(1.0 / timeseries.delta_t) / 2))
    coeff = signal.firwin(order * 2 + 1, k, window=('kaiser', beta))
    return fir_zero_filter(coeff, timeseries)


def highpass_fir(timeseries, frequency, order, beta=5.0):
    """ Highpass filter the time series using an FIR filtered generated from
    the ideal response passed through a kaiser window (beta = 5.0)

    Parameters
    ----------
    Time Series: TimeSeries
        The time series to be high-passed.
    frequency: float
        The frequency below which is suppressed.
    order: int
        Number of corrupted samples on each side of the time series
    beta: float
        Beta parameter of the kaiser window that sets the side lobe attenuation.
    """
    k = frequency / float((int(1.0 / timeseries.delta_t) / 2))
    coeff = signal.firwin(order * 2 + 1, k, window=('kaiser', beta), pass_zero=False)
    return fir_zero_filter(coeff, timeseries)

# =============================================================================
# 
# =============================================================================
def time_slice(timeseries, start, end, mode='floor'):
    """Return the slice of the time series that contains the time range
    in GPS seconds.
    """
    # if start < timeseries.start_time:
    #     raise ValueError('Time series does not contain a time as early as %s' % start)

    # if end > timeseries.end_time:
    #     raise ValueError('Time series does not contain a time as late as %s' % end)
    
    start_idx = float(start) * timeseries.sample_rate
    end_idx = fload(end) * timeseries.sample_rate
    
    # start_idx = float(start - self.start_time) * self.sample_rate
    # end_idx = float(end - self.start_time) * self.sample_rate

    if _numpy.isclose(start_idx, round(start_idx)):
        start_idx = round(start_idx)

    if _numpy.isclose(end_idx, round(end_idx)):
        end_idx = round(end_idx)

    if mode == 'floor':
        start_idx = int(start_idx)
        end_idx = int(end_idx)
    elif mode == 'nearest':
        start_idx = int(round(start_idx))
        end_idx = int(round(end_idx))
    else:
        raise ValueError("Invalid mode: {}".format(mode))

    return timeseries[start_idx:end_idx]
