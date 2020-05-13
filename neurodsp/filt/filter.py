"""Filter time series."""

from warnings import warn

from neurodsp.filt.fir import filter_signal_fir
from neurodsp.filt.iir import filter_signal_iir

###################################################################################################
###################################################################################################

def filter_signal(sig, fs, pass_type, f_range, filter_type='fir',
                  n_cycles=3, n_seconds=None, remove_edges=True, butterworth_order=None,
                  print_transitions=False, plot_properties=False, return_filter=False):
    """Apply a bandpass, bandstop, highpass, or lowpass filter to a neural signal.

    Parameters
    ----------
    sig : 1d array
        Time series to be filtered.
    fs : float
        Sampling rate, in Hz.
    pass_type : {'bandpass', 'bandstop', 'lowpass', 'highpass'}
        Which kind of filter to apply:

        * 'bandpass': apply a bandpass filter
        * 'bandstop': apply a bandstop (notch) filter
        * 'lowpass': apply a lowpass filter
        * 'highpass' : apply a highpass filter
    f_range : tuple of (float, float) or float
        Cutoff frequency(ies) used for filter, specified as f_lo & f_hi.
        For 'bandpass' & 'bandstop', must be a tuple.
        For 'lowpass' or 'highpass', can be a float that specifies pass frequency, or can be
        a tuple and is assumed to be (None, f_hi) for 'lowpass', and (f_lo, None) for 'highpass'.
    n_cycles : float, optional, default: 3
        Length of filter, in number of cycles, at the 'f_lo' frequency, if using an FIR filter.
        This parameter is overwritten by `n_seconds`, if provided.
    n_seconds : float, optional
        Length of filter, in seconds, if using an FIR filter.
        This parameter overwrites `n_cycles`.
    filter_type : {'fir', 'iir'}, optional
        Whether to use an FIR or IIR filter.
        The only IIR filter offered is a butterworth filter.
    remove_edges : bool, optional, default: True
        If True, replace samples within half the kernel length to be np.nan.
        Only used for FIR filters.
    butterworth_order : int, optional
        Order of the butterworth filter, if using an IIR filter.
        See input 'N' in scipy.signal.butter.
    print_transitions : bool, optional, default: True
        If True, print out the transition and pass bandwidths.
    plot_properties : bool, optional, default: False
        If True, plot the properties of the filter, including frequency response and/or kernel.
    return_filter : bool, optional, default: False
        If True, return the filter coefficients.

    Returns
    -------
    sig_filt : 1d array
        Filtered time series.
    kernel : 1d array or tuple of (1d array, 1d array)
        Filter coefficients. Only returned if `return_filter` is True.

    Examples
    --------
    Apply an FIR band pass filter to a signal, for the range of 1 to 25 Hz:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> filt_sig = filter_signal(sig, fs=500, pass_type='bandpass',
    ...                          filter_type='fir', f_range=(1, 25))
    """

    if filter_type.lower() == 'fir':
        return filter_signal_fir(sig, fs, pass_type, f_range, n_cycles, n_seconds,
                                 remove_edges, print_transitions,
                                 plot_properties, return_filter)
    elif filter_type.lower() == 'iir':
        _iir_checks(n_seconds, butterworth_order, remove_edges)
        return filter_signal_iir(sig, fs, pass_type, f_range, butterworth_order,
                                 print_transitions, plot_properties,
                                 return_filter)
    else:
        raise ValueError('Filter type not understood.')


def _iir_checks(n_seconds, butterworth_order, remove_edges):
    """Checks for using an IIR filter if called from the general filter function."""

    # Check inputs for IIR filters
    if n_seconds is not None:
        raise ValueError('n_seconds should not be defined for an IIR filter.')
    if butterworth_order is None:
        raise ValueError('butterworth_order must be defined when using an IIR filter.')
    if remove_edges:
        warn('Edge artifacts are not removed when using an IIR filter.')
