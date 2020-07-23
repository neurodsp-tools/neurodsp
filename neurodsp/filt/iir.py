"""Filter signals with IIR filters."""

from scipy.signal import butter, sosfiltfilt

from neurodsp.utils import remove_nans, restore_nans
from neurodsp.filt.utils import compute_nyquist, compute_frequency_response
from neurodsp.filt.checks import check_filter_definition, check_filter_properties
from neurodsp.plts.filt import plot_frequency_response

###################################################################################################
###################################################################################################

def filter_signal_iir(sig, fs, pass_type, f_range, butterworth_order,
                      print_transitions=False, plot_properties=False, return_filter=False):
    """Apply an IIR filter to a signal.

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
    butterworth_order : int
        Order of the butterworth filter, if using an IIR filter.
        See input 'N' in scipy.signal.butter.
    print_transitions : bool, optional, default: False
        If True, print out the transition and pass bandwidths.
    plot_properties : bool, optional, default: False
        If True, plot the properties of the filter, including frequency response and/or kernel.
    return_filter : bool, optional, default: False
        If True, return the second order series coefficients of the IIR filter.

    Returns
    -------
    sig_filt : 1d array
        Filtered time series.
    sos : 2d array
        Second order series coefficients of the IIR filter. Has shape of (n_sections, 6).
        Only returned if `return_filter` is True.

    Examples
    --------
    Apply a bandstop IIR filter to a simulated signal:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> filt_sig = filter_signal_iir(sig, fs=500, pass_type='bandstop',
    ...                              f_range=(55, 65), butterworth_order=7)
    """

    # Design filter
    sos = design_iir_filter(fs, pass_type, f_range, butterworth_order)

    # Check filter properties: compute transition bandwidth & run checks
    check_filter_properties(sos, None, fs, pass_type, f_range, verbose=print_transitions)

    # Remove any NaN on the edges of 'sig'
    sig, sig_nans = remove_nans(sig)

    # Apply filter
    sig_filt = apply_iir_filter(sig, sos)

    # Add NaN back on the edges of 'sig', if there were any at the beginning
    sig_filt = restore_nans(sig_filt, sig_nans)

    # Plot frequency response, if desired
    if plot_properties:
        f_db, db = compute_frequency_response(sos, None, fs)
        plot_frequency_response(f_db, db)

    if return_filter:
        return sig_filt, sos
    else:
        return sig_filt


def apply_iir_filter(sig, sos):
    """Apply an IIR filter to a signal.

    Parameters
    ----------
    sig : array
        Time series to be filtered.
    sos : 2d array
        Second order series coefficients for an IIR filter. Has shape of (n_sections, 6).

    Returns
    -------
    array
        Filtered time series.

    Examples
    --------
    Apply an IIR filter, after designing the filter coefficients:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> sos = design_iir_filter(fs=500, pass_type='bandstop',
    ...                                    f_range=(55, 65), butterworth_order=7)
    >>> filt_signal = apply_iir_filter(sig, sos)
    """

    return sosfiltfilt(sos, sig)


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
    sos : 2d array
        Second order series coefficients for an IIR filter. Has shape of (n_sections, 6).

    Examples
    --------
    Compute coefficients for a bandstop IIR filter:

    >>> sos = design_iir_filter(fs=500, pass_type='bandstop',
    ...                         f_range=(55, 65), butterworth_order=7)
    """

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
    sos = butter(butterworth_order, win, pass_type, output='sos')

    return sos
