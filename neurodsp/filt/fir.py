"""Filter signals with FIR filters."""

import numpy as np
from scipy.signal import firwin

from neurodsp.utils import remove_nans, restore_nans
from neurodsp.utils.decorators import multidim
from neurodsp.plts.filt import plot_filter_properties
from neurodsp.filt.utils import compute_frequency_response, remove_filter_edges
from neurodsp.filt.checks import (check_filter_definition, check_filter_properties,
                                  check_filter_length)

###################################################################################################
###################################################################################################

def filter_signal_fir(sig, fs, pass_type, f_range, n_cycles=3, n_seconds=None, remove_edges=True,
                      print_transitions=False, plot_properties=False, return_filter=False):
    """Apply an FIR filter to a signal.

    Parameters
    ----------
    sig : array
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
        Length of filter, in number of cycles, defined at the 'f_lo' frequency.
        This parameter is overwritten by `n_seconds`, if provided.
    n_seconds : float, optional
        Length of filter, in seconds. This parameter overwrites `n_cycles`.
    remove_edges : bool, optional
        If True, replace samples within half the kernel length to be np.nan.
    print_transitions : bool, optional, default: False
        If True, print out the transition and pass bandwidths.
    plot_properties : bool, optional, default: False
        If True, plot the properties of the filter, including frequency response and/or kernel.
    return_filter : bool, optional, default: False
        If True, return the filter coefficients of the FIR filter.

    Returns
    -------
    sig_filt : array
        Filtered time series.
    filter_coefs : 1d array
        Filter coefficients of the FIR filter. Only returned if `return_filter` is True.

    Examples
    --------
    Apply a band pass FIR filter to a simulated signal:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> filt_sig = filter_signal_fir(sig, fs=500, pass_type='bandpass', f_range=(1, 25))

    Apply a high pass FIR filter to a signal, with a specified number of cycles:

    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> filt_sig = filter_signal_fir(sig, fs=500, pass_type='highpass',
    ...                              f_range=(2, None), n_cycles=5)
    """

    # Design filter & check that the length is okay with signal
    filter_coefs = design_fir_filter(fs, pass_type, f_range, n_cycles, n_seconds)
    check_filter_length(sig.shape[-1], len(filter_coefs))

    # Check filter properties: compute transition bandwidth & run checks
    check_filter_properties(filter_coefs, 1, fs, pass_type, f_range, verbose=print_transitions)

    # Remove any NaN on the edges of 'sig'
    sig, sig_nans = remove_nans(sig)

    # Apply filter
    sig_filt = apply_fir_filter(sig, filter_coefs)

    # Remove edge artifacts
    if remove_edges:
        sig_filt = remove_filter_edges(sig_filt, len(filter_coefs))

    # Add NaN back on the edges of 'sig', if there were any at the beginning
    sig_filt = restore_nans(sig_filt, sig_nans)

    # Plot filter properties, if specified
    if plot_properties:
        f_db, db = compute_frequency_response(filter_coefs, 1, fs)
        plot_filter_properties(f_db, db, fs, filter_coefs)

    if return_filter:
        return sig_filt, filter_coefs
    else:
        return sig_filt


@multidim()
def apply_fir_filter(sig, filter_coefs):
    """Apply an FIR filter to a signal.

    Parameters
    ----------
    sig : array
        Time series to be filtered.
    filter_coefs : 1d array
        Filter coefficients of the FIR filter.

    Returns
    -------
    array
        Filtered time series.

    Examples
    --------
    Apply an FIR filter, from computed filter coefficients:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> filter_coefs = design_fir_filter(fs=500, pass_type='bandpass', f_range=(1, 25))
    >>> filt_sig = apply_fir_filter(sig, filter_coefs)
    """

    return np.convolve(filter_coefs, sig, 'same')


def design_fir_filter(fs, pass_type, f_range, n_cycles=3, n_seconds=None):
    """Design an FIR filter.

    Parameters
    ----------
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
        Length of filter, in number of cycles, defined at the 'f_lo' frequency.
        This parameter is overwritten by `n_seconds`, if provided.
    n_seconds : float or None, optional
        Length of filter, in seconds. This parameter overwrites `n_cycles`.

    Returns
    -------
    filter_coefs : 1d array
        The filter coefficients for an FIR filter.

    Examples
    --------
    Create the filter coefficients for an FIR filter:

    >>> filter_coefs = design_fir_filter(fs=500, pass_type='bandpass', f_range=(1, 25))
    """

    # Check filter definition
    f_lo, f_hi = check_filter_definition(pass_type, f_range)
    filt_len = compute_filter_length(fs, pass_type, f_lo, f_hi, n_cycles, n_seconds)

    if pass_type == 'bandpass':
        filter_coefs = firwin(filt_len, (f_lo, f_hi), pass_zero=False, fs=fs)
    elif pass_type == 'bandstop':
        filter_coefs = firwin(filt_len, (f_lo, f_hi), fs=fs)
    elif pass_type == 'highpass':
        filter_coefs = firwin(filt_len, f_lo, pass_zero=False, fs=fs)
    elif pass_type == 'lowpass':
        filter_coefs = firwin(filt_len, f_hi, fs=fs)

    return filter_coefs


def compute_filter_length(fs, pass_type, f_lo, f_hi, n_cycles=None, n_seconds=None):
    """Compute the filter length for an FIR signal given specified parameters.

    Parameters
    ----------
    fs : float
        Sampling rate, in Hz.
    pass_type : {'bandpass', 'bandstop', 'lowpass', 'highpass'}
        Which kind of filter to apply.
    f_lo : float or None
        The lower frequency range of the filter, specifying the highpass frequency, if specified.
    f_hi : float or None
        The higher frequency range of the filter, specifying the lowpass frequency, if specified.
    n_cycles : float or None, optional
        Length of filter, in number of cycles, defined at the 'f_lo' frequency.
    n_seconds : float or None, optional
        Length of filter, in seconds.

    Returns
    -------
    filt_len : int
        The length of the specified filter.

    Examples
    --------
    Compute the length of bandpass (1 to 25 Hz) filter:

    >>> filt_len = compute_filter_length(fs=500, pass_type='bandpass', f_lo=1, f_hi=25, n_cycles=3)
    """

    # Compute filter length if specified in seconds
    if n_seconds is not None:
        filt_len = fs * n_seconds
    # Otherwise, calculate filter length based on number of cycles
    elif n_cycles is not None:
        if pass_type == 'lowpass':
            filt_len = fs * n_cycles / f_hi
        else:
            filt_len = fs * n_cycles / f_lo
    else:
        raise ValueError('Either `n_cycles` or `n_seconds` needs to be defined.')

    # Typecast filter length to an integer, rounding up & force length to be odd
    filt_len = int(np.ceil(filt_len))
    if filt_len % 2 == 0:
        filt_len = filt_len + 1

    return filt_len
