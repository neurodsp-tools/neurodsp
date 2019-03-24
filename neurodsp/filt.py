"""Filter a neural signal using a bandpass, highpass, lowpass, or bandstop filter."""

import warnings

import numpy as np
from scipy import signal

from neurodsp.utils import remove_nans, restore_nans
from neurodsp.plts.filt import plot_filter_properties, plot_frequency_response

###################################################################################################
###################################################################################################

def filter_signal(sig, fs, pass_type, f_range, n_cycles=3, n_seconds=None,
                  filt_type='fir', remove_edges=True, butterworth_order=None,
                  print_transitions=True, plot_properties=False, return_filter=False):
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
        Length of filter, in number of cycles, defined at the 'f_lo' frequency.
        This parameter is overwritten by `n_seconds`, if provided.
    n_seconds : float, optional
        Length of filter, in seconds. This parameter overwrites `n_cycles`.
    filt_type : {'fir', 'iir'}, optional
        Whether to use an FIR or IIR filter.
        The only IIR filter offered is a butterworth filter.
    remove_edges : bool, optional, default: True
        If True, replace samples within half the kernel length to be np.nan.
        Only used for FIR filters.
    butterworth_order : int, optional
        Order of the butterworth filter, if using an IIR filter.
        See input 'N' in scipy.signal.butter.
    print_transitions : bool, optional, default: True
        If True, computes the transition bandwidth(s), and prints this information.
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
    """

    if filt_type == 'fir':
        return filter_signal_fir(sig, fs, pass_type, f_range, n_cycles, n_seconds,
                                 remove_edges, print_transitions,
                                 plot_properties, return_filter)
    elif filt_type == 'iir':
        _iir_checks(n_seconds, butterworth_order, remove_edges)
        return filter_signal_iir(sig, fs, pass_type, f_range, butterworth_order,
                                 print_transitions, plot_properties,
                                 return_filter)
    else:
        raise ValueError('Filter type not understood.')


def filter_signal_fir(sig, fs, pass_type, f_range, n_cycles=3, n_seconds=None, remove_edges=True,
                      print_transitions=True, plot_properties=False, return_filter=False):
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
    print_transitions : bool, optional
        If True, computes the transition bandwidth, and prints this information.
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

    # Compute transition bandwidth
    if print_transitions:
        check_filter_properties(filter_coefs, 1, fs, pass_type, f_range)

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


def filter_signal_iir(sig, fs, pass_type, f_range, butterworth_order, print_transitions=True,
                      plot_properties=False, return_filter=False):
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
    print_transitions : bool, optional, default: True
        If True, computes the transition bandwidth, and prints this information.
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

    # Compute transition bandwidth
    if print_transitions:
        check_filter_properties(b_vals, a_vals, fs, pass_type, f_range)

    # Remove any NaN on the edges of 'sig'
    sig, sig_nans = remove_nans(sig)

    # Apply filter
    sig_filt = signal.filtfilt(b_vals, a_vals, sig)

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
    filt_len = _fir_checks(pass_type, f_lo, f_hi, n_cycles, n_seconds, fs, sig_length)

    f_nyq = compute_nyquist(fs)
    if pass_type == 'bandpass':
        filter_coefs = signal.firwin(filt_len, (f_lo, f_hi), pass_zero=False, nyq=f_nyq)
    elif pass_type == 'bandstop':
        filter_coefs = signal.firwin(filt_len, (f_lo, f_hi), nyq=f_nyq)
    elif pass_type == 'highpass':
        filter_coefs = signal.firwin(filt_len, f_lo, pass_zero=False, nyq=f_nyq)
    elif pass_type == 'lowpass':
        filter_coefs = signal.firwin(filt_len, f_hi, nyq=f_nyq)

    return filter_coefs


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
        warnings.warn('IIR filters are not recommended other than for notch filters.')

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
    b_vals, a_vals = signal.butter(butterworth_order, win, pass_type)

    return b_vals, a_vals


def check_filter_definition(pass_type, f_range):
    """Check a filter definition for validity, and get f_lo and f_hi.

    Parameters
    ----------
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

    Returns
    -------
    f_lo : float or None
        The lowpass frequency of the filter, if specified.
    f_hi : float or None
        The highpass frequency of the filter, if specified.
    """

    if pass_type not in ['bandpass', 'bandstop', 'lowpass', 'highpass']:
        raise ValueError('Filter passtype not understood.')

    ## Check that frequency cutoff inputs are appropriate
    # For band filters, 2 inputs required & second entry must be > first
    if pass_type in ('bandpass', 'bandstop'):
        if isinstance(f_range, tuple) and f_range[0] >= f_range[1]:
            raise ValueError('Second cutoff frequency must be greater than first.')
        elif isinstance(f_range, (int, float)) or len(f_range) != 2:
            raise ValueError('Two cutoff frequencies required for bandpass and bandstop filters')

        # Map f_range to f_lo and f_hi
        f_lo, f_hi = f_range

    # For lowpass and highpass can be tuple or int/float
    if pass_type == 'lowpass':
        if isinstance(f_range, (int, float)):
            f_hi = f_range
        elif isinstance(f_range, tuple):
            f_hi = f_range[1]
        f_lo = None

    if pass_type == 'highpass':
        if isinstance(f_range, (int, float)):
            f_lo = f_range
        elif isinstance(f_range, tuple):
            f_lo = f_range[0]
        f_hi = None

    # Make sure pass freqs are floats
    f_lo = float(f_lo) if f_lo else f_lo
    f_hi = float(f_hi) if f_hi else f_hi

    return f_lo, f_hi


def check_filter_properties(b_vals, a_vals, fs, pass_type, f_range, transitions=(-20, -3)):
    """Check a filters properties, including pass band and transition band.

    Parameters
    ----------
    b_vals : 1d array
        B value filter coefficients for a filter.
    a_vals : 1d array
        A value filter coefficients for a filter.
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
    transitions : tuple of (float, float), optional, default: (-20, -3)
        Cutoffs, in dB, that define the transition band.
    """

    # Compute the frequency response
    f_db, db = compute_frequency_response(b_vals, a_vals, fs)

    # Check that frequency response goes below transition level (has significant attenuation)
    if np.min(db) >= transitions[0]:
        warnings.warn('The filter attenuation never goes below {} dB.'\
                      'Increase filter length.'.format(transitions[0]))
        # If there is no attenuation, cannot calculate bands - so return here
        return

    # Check that both sides of a bandpass have significant attenuation
    if pass_type == 'bandpass':
        if db[0] >= transitions[0] or db[-1] >= transitions[0]:
            warnings.warn('The low or high frequency stopband never gets attenuated by'\
                          'more than {} dB. Increase filter length.'.format(abs(transitions[0])))

    # Compute pass & transition bandwidth
    pass_bw = compute_pass_band(fs, pass_type, f_range)
    transition_bw = compute_transition_band(f_db, db, transitions[0], transitions[1])

    # Raise warning if transition bandwidth is too high
    if transition_bw > pass_bw:
        warnings.warn('Transition bandwidth is  {:.1f}  Hz. This is greater than the desired'\
                      'pass/stop bandwidth of  {:.1f} Hz'.format(transition_bw, pass_bw))

    # Print out transition bandwidth and pass bandwidth to the user
    print('Transition bandwidth is {:.1f} Hz.'.format(transition_bw))
    print('Pass/stop bandwidth is {:.1f} Hz.'.format(pass_bw))


def compute_frequency_response(b_vals, a_vals, fs):
    """Compute the frequency response of a filter.

    Parameters
    ----------
    b_vals : 1d array
        B value filter coefficients for a filter.
    a_vals : 1d array
        A value filter coefficients for a filter.
    fs : float
        Sampling rate, in Hz.

    Returns
    -------
    f_db : 1d array
        Frequency vector corresponding to attenuation decibels, in Hz.
    db : 1d array
        Degree of attenuation for each frequency specified in f_db, in dB.
    """

    w_vals, h_vals = signal.freqz(b_vals, a_vals)
    f_db = w_vals * fs / (2. * np.pi)
    db = 20 * np.log10(abs(h_vals))

    return f_db, db


def compute_pass_band(fs, pass_type, f_range):
    """Compute the pass bandwidth of a filter.

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

    Returns
    -------
    pass_bw : float
        The pass bandwidth of the filter.
    """

    f_lo, f_hi = check_filter_definition(pass_type, f_range)
    if pass_type in ['bandpass', 'bandstop']:
        pass_bw = f_hi - f_lo
    elif pass_type == 'highpass':
        pass_bw = compute_nyquist(fs) - f_lo
    elif pass_type == 'lowpass':
        pass_bw = f_hi

    return pass_bw


def compute_transition_band(f_db, db, low=-20, high=-3):
    """Compute transition bandwidth of a filter.

    Parameters
    ----------
    f_db : 1d array
        Frequency vector corresponding to attenuation decibels, in Hz.
    db : 1d array
        Degree of attenuation for each frequency specified in f_db, in dB.
    low : float, optional, default: -20
        The lower limit that defines the transition band, in dB.
    high : float, optional, default: -3
        The upper limit that defines the transition band, in dB.

    Returns
    -------
    transition_band : float
        The transition bandwidth of the filter.
    """

    # This gets the indices of transitions to the values in searched for range
    inds = np.where(np.diff(np.logical_and(db > low, db < high)))[0]
    # This steps through the indices, in pairs, selecting from the vector to select from
    transition_band = np.max([(b - a) for a, b in zip(f_db[inds[0::2]], f_db[inds[1::2]])])

    return transition_band


def compute_nyquist(fs):
    """Compute the nyquist frequency.

    Parameters
    ----------
    fs : float
        Sampling rate, in Hz.

    Returns
    -------
    float
        The nyquist frequency of a signal with the given sampling rate, in Hz.
    """

    return fs / 2.


def infer_passtype(f_range):
    """Given frequency definition of a filter, infer the passtype.

    Parameters
    ----------
    f_range : tuple of (float, float)
        Cutoff frequency(ies) used for filter, specified as f_lo & f_hi.

    Returns
    -------
    pass_type : str
        Which kind of filter pass_type is consistent with the frequency definition provided.

    Notes
    -----
    Assumes that a definition with two frequencies is a 'bandpass' (not 'bandstop').
    """

    if f_range[0] is None:
        pass_type = 'lowpass'
    elif f_range[1] is None:
        pass_type = 'highpass'
    else:
        pass_type = 'bandpass'

    # Check the inferred passtype & frequency definition is valid
    _ = check_filter_definition(pass_type, f_range)

    return pass_type


def remove_filter_edges(sig, filt_len):
    """Drop the edges, by making NaN, from a filtered signal, to avoid edge artifacts.

    Parameters
    ----------
    sig : 1d array
        Filtered signal to have edge artifacts removed from.
    filt_len : int
        Length of the filter that was applied.

    Returns
    -------
    sig : 1d array
        Filter signal with edge artifacts switched to NaNs.
    """

    n_rmv = int(np.ceil(filt_len / 2))
    sig[:n_rmv] = np.nan
    sig[-n_rmv:] = np.nan

    return sig

###################################################################################################
###################################################################################################

def _fir_checks(pass_type, f_lo, f_hi, n_cycles, n_seconds, fs, sig_length):
    """Check for running an FIR filter, including figuring out the filter length."""

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


def _iir_checks(n_seconds, butterworth_order, remove_edges):
    """Checks for using an IIR filter if called from the general filter function."""

    # Check inputs for IIR filters
    if n_seconds is not None:
        raise ValueError('n_seconds should not be defined for an IIR filter.')
    if butterworth_order is None:
        raise ValueError('butterworth_order must be defined when using an IIR filter.')
    if remove_edges:
        warnings.warn('Edge artifacts are not removed when using an IIR filter.')
