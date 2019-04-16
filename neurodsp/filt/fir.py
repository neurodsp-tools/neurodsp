"""Filter signals with FIR filters."""

import numpy as np
from scipy.signal import firwin

from neurodsp.utils import remove_nans, restore_nans
from neurodsp.utils.decorators import multidim
from neurodsp.plts.filt import plot_filter_properties
from neurodsp.filt.utils import compute_nyquist, compute_frequency_response, remove_filter_edges
from neurodsp.filt.checks import check_filter_definition, check_filter_properties

###################################################################################################
###################################################################################################

@multidim
def filter_signal_fir(sig, fs, pass_type, f_range, n_cycles=3, n_seconds=None, remove_edges=True,
                      print_transitions=False, plot_properties=False, return_filter=False):
    """Apply an FIR filter to a signal.

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
    sig_filt : 1d array
        Filtered time series.
    filter_coefs : 1d array
        Filter coefficients of the FIR filter. Only returned if `return_filter` is True.
    """

    # Design filter
    filter_coefs = design_fir_filter(len(sig), fs, pass_type, f_range, n_cycles, n_seconds)

    # Check filter properties: compute transition bandwidth & run checks
    check_filter_properties(filter_coefs, 1, fs, pass_type, f_range, verbose=print_transitions)

    # Remove any NaN on the edges of 'sig'
    sig, sig_nans = remove_nans(sig)

    # Apply filter
    sig_filt = np.convolve(filter_coefs, sig, 'same')

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


def design_fir_filter(sig_length, fs, pass_type, f_range, n_cycles=3, n_seconds=None):
    """Design an FIR filter.

    Parameters
    ----------
    sig_length : int
        The length of the signal to be filtered.
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

    Returns
    -------
    filter_coefs : 1d array
        The filter coefficients for an FIR filter.
    """

    # Check filter definition
    f_lo, f_hi = check_filter_definition(pass_type, f_range)
    filt_len = compute_filt_len(sig_length, fs, pass_type, f_lo, f_hi, n_cycles, n_seconds)

    f_nyq = compute_nyquist(fs)
    if pass_type == 'bandpass':
        filter_coefs = firwin(filt_len, (f_lo, f_hi), pass_zero=False, nyq=f_nyq)
    elif pass_type == 'bandstop':
        filter_coefs = firwin(filt_len, (f_lo, f_hi), nyq=f_nyq)
    elif pass_type == 'highpass':
        filter_coefs = firwin(filt_len, f_lo, pass_zero=False, nyq=f_nyq)
    elif pass_type == 'lowpass':
        filter_coefs = firwin(filt_len, f_hi, nyq=f_nyq)

    return filter_coefs


def compute_filt_len(sig_length, fs, pass_type, f_lo, f_hi, n_cycles, n_seconds):
    """Calculate and check the filter length for an FIR signal with specified parameters.

    Parameters
    ----------
    sig_length : int
        The length of the signal to be filtered.
    fs : float
        Sampling rate, in Hz.
    pass_type : {'bandpass', 'bandstop', 'lowpass', 'highpass'}
        Which kind of filter to apply.
    f_lo : float or None
        The lower frequency range of the filter, specifying the highpass frequency, if specified.
    f_hi : float or None
        The higher frequency range of the filter, specifying the lowpass frequency, if specified.
    n_cycles : float, optional, default: 3
        Length of filter, in number of cycles, defined at the 'f_lo' frequency.
    n_seconds : float, optional
        Length of filter, in seconds.

    Returns
    -------
    filt_len : int
        The length of the specified filter.
    """

    # Compute filter length if specified in seconds
    if n_seconds is not None:
        filt_len = fs * n_seconds
    else:
        if pass_type == 'lowpass':
            filt_len = fs * n_cycles / f_hi
        else:
            filt_len = fs * n_cycles / f_lo

    # Typecast filter length to an integer, rounding up
    filt_len = int(np.ceil(filt_len))

    # Force filter length to be odd
    if filt_len % 2 == 0:
        filt_len = filt_len + 1

    # Raise an error if the filter is longer than the signal
    if filt_len >= sig_length:
        raise ValueError(
            'The designed filter (length: {:d}) is longer than the signal '\
            '(length: {:d}). The filter needs to be shortened by decreasing '\
            'the n_cycles or n_seconds parameter. However, this will decrease '\
            'the frequency resolution of the filter.'.format(filt_len, sig_length))

    return filt_len
