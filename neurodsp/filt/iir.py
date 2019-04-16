"""Filter signals with IIR filters."""

from warnings import warn

from scipy.signal import butter, filtfilt

from neurodsp.utils import remove_nans, restore_nans
from neurodsp.utils.decorators import multidim
from neurodsp.filt.utils import compute_nyquist, compute_frequency_response
from neurodsp.filt.checks import check_filter_definition, check_filter_properties
from neurodsp.plts.filt import plot_frequency_response

###################################################################################################
###################################################################################################

@multidim
def filter_signal_iir(sig, fs, pass_type, f_range, butterworth_order,
                      print_transitions=False, plot_properties=False, return_filter=False):
    """Apply an IIR filter to a signal.

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
    butterworth_order : int
        Order of the butterworth filter, if using an IIR filter.
        See input 'N' in scipy.signal.butter.
    print_transitions : bool, optional, default: False
        If True, print out the transition and pass bandwidths.
    plot_properties : bool, optional, default: False
        If True, plot the properties of the filter, including frequency response and/or kernel.
    return_filter : bool, optional, default: False
        If True, return the filter coefficients of the IIR filter.

    Returns
    -------
    sig_filt : 1d array
        Filtered time series.
    filter_coefs : tuple of (1d array, 1d array)
        Filter coefficients of the IIR filter, as (b_vals, a_vals).
        Only returned if `return_filter` is True.
    """

    # Design filter
    b_vals, a_vals = design_iir_filter(fs, pass_type, f_range, butterworth_order)

    # Check filter properties: compute transition bandwidth & run checks
    check_filter_properties(b_vals, a_vals, fs, pass_type, f_range, verbose=print_transitions)

    # Remove any NaN on the edges of 'sig'
    sig, sig_nans = remove_nans(sig)

    # Apply filter
    sig_filt = filtfilt(b_vals, a_vals, sig)

    # Add NaN back on the edges of 'sig', if there were any at the beginning
    sig_filt = restore_nans(sig_filt, sig_nans)

    # Plot frequency response, if desired
    if plot_properties:
        f_db, db = compute_frequency_response(b_vals, a_vals, fs)
        plot_frequency_response(f_db, db)

    if return_filter:
        return sig_filt, (b_vals, a_vals)
    else:
        return sig_filt


def design_iir_filter(fs, pass_type, f_range, butterworth_order):
    """Design an IIR filter.

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
    butterworth_order : int
        Order of the butterworth filter, if using an IIR filter.
        See input 'N' in scipy.signal.butter.

    Returns
    -------
    b_vals : 1d array
        B value filter coefficients for an IIR filter.
    a_vals : 1d array
        A value filter coefficients for an IIR filter.
    """

    # Warn about only recommending IIR for bandstop
    if pass_type != 'bandstop':
        warn('IIR filters are not recommended other than for notch filters.')

    # Check filter definition
    f_lo, f_hi = check_filter_definition(pass_type, f_range)

    f_nyq = compute_nyquist(fs)
    if pass_type in ('bandpass', 'bandstop'):
        win = (f_lo / f_nyq, f_hi / f_nyq)
    elif pass_type == 'highpass':
        win = f_lo / f_nyq
    elif pass_type == 'lowpass':
        win = f_hi / f_nyq

    # Design filter
    b_vals, a_vals = butter(butterworth_order, win, pass_type)

    return b_vals, a_vals
