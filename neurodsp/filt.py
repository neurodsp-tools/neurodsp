"""Filter a neural signal using a bandpass, highpass, lowpass, or bandstop filter."""

import warnings

import numpy as np
from scipy import signal

from neurodsp.plts.filt import plot_frequency_response

###################################################################################################
###################################################################################################

def filter_signal(sig, fs, pass_type, fc, n_cycles=3, n_seconds=None,
                  iir=False, butterworth_order=None,
                  plot_freq_response=False, return_kernel=False,
                  verbose=False, compute_transition_band=True, remove_edge_artifacts=True):
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
    fc : tuple of (float, float) or float
        Cutoff frequency(ies) used for filter, specified as f_lo & f_hi.
        For 'bandpass' & 'bandstop', must be a tuple.
        For 'lowpass' or 'highpass', can be a float that specifies pass frequency, or can be
        a tuple and is assumed to be (None, f_hi) for 'lowpass', and (f_lo, None) for 'highpass'.
    n_cycles : float, optional, default: 3
        Length of filter, in number of cycles, defined at the 'f_lo' frequency.
        This parameter is overwritten by `n_seconds`, if provided.
    n_seconds : float, optional
        Length of filter, in seconds.
        This parameter overwrites `n_cycles`.
    iir : bool, optional
        If True, use an infinite-impulse response (IIR) filter.
        The only IIR filter offered is a butterworth filter.
    butterworth_order : int, optional
        Order of the butterworth filter, if using an IIR filter.
        See input 'N' in scipy.signal.butter.
    plot_freq_response : bool, optional, default: False
        If True, plot the frequency response of the filter
    return_kernel : bool, optional, default: False
        If True, return the complex filter kernel.
    verbose : bool, optional, default: False
        If True, print optional information.
    compute_transition_band : bool, optional, default: True
        If True, computes the transition bandwidth, and prints this information.
    remove_edge_artifacts : bool, optional, default: True
        If True, replace samples within half the kernel length to be np.nan.

    Returns
    -------
    sig_filt : 1d array
        Filtered time series.
    kernel : length-2 tuple of arrays
        Filter kernel. Only returned if `return_kernel` is True.
    """

    if iir:
        _iir_checks(n_seconds, butterworth_order, remove_edge_artifacts)
        return filter_signal_iir(sig, fs, pass_type, fc, butterworth_order, plot_freq_response,
                                 return_kernel, compute_transition_band)
    else:
        return filter_signal_fir(sig, fs, pass_type, fc, n_cycles, n_seconds, plot_freq_response,
                                 return_kernel, compute_transition_band, remove_edge_artifacts)


def filter_signal_fir(sig, fs, pass_type, fc, n_cycles=3, n_seconds=None, plot_freq_response=False,
                      return_kernel=False, compute_transition_band=True, remove_edge_artifacts=True):
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
    fc : tuple of (float, float) or float
        Cutoff frequency(ies) used for filter, specified as f_lo & f_hi.
        For 'bandpass' & 'bandstop', must be a tuple.
        For 'lowpass' or 'highpass', can be a float that specifies pass frequency, or can be
        a tuple and is assumed to be (None, f_hi) for 'lowpass', and (f_lo, None) for 'highpass'.
    n_cycles : float, optional, default: 3
        Length of filter, in number of cycles, defined at the 'f_lo' frequency.
        This parameter is overwritten by `n_seconds`, if provided.
    n_seconds : float, optional
        Length of filter, in seconds.
        This parameter overwrites `n_cycles`.
    plot_freq_response : bool, optional, default: False
        If True, plot the frequency response of the filter
    return_kernel : bool, optional, default: False
        If True, return the complex filter kernel.
    compute_transition_band : bool, optional
        If True, computes the transition bandwidth, and prints this information.
    remove_edge_artifacts : bool, optional
        If True, replace samples within half the kernel length to be np.nan.

    Returns
    -------
    sig_filt : 1d array
        Filtered time series.
    kernel : length-2 tuple of arrays
        Filter kernel. Only returned if `return_kernel` is True.
    """

    # Design filter
    kernel = design_fir_filter(len(sig), fs, pass_type, fc, n_cycles, n_seconds)

    # Compute transition bandwidth
    if compute_transition_band:
        a_vals, b_vals = 1, kernel
        check_filter_properties(b_vals, a_vals, fs, pass_type, fc)

    # Remove any NaN on the edges of 'sig'
    sig, sig_nans = _remove_nans(sig)

    # Apply filter
    sig_filt = np.convolve(kernel, sig, 'same')

    # Remove edge artifacts
    if remove_edge_artifacts:
        sig_filt = _drop_edge_artifacts(sig_filt, len(kernel))

    # Add NaN back on the edges of 'sig', if there were any at the beginning
    sig_filt = _restore_nans(sig_filt, sig_nans)

    # Plot frequency response, if desired
    if plot_freq_response:
        plot_frequency_response(fs, kernel)

    if return_kernel:
        return sig_filt, kernel
    else:
        return sig_filt


def filter_signal_iir(sig, fs, pass_type, fc, butterworth_order, plot_freq_response=False,
                      return_kernel=False, compute_transition_band=True):
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
    fc : tuple of (float, float) or float
        Cutoff frequency(ies) used for filter, specified as f_lo & f_hi.
        For 'bandpass' & 'bandstop', must be a tuple.
        For 'lowpass' or 'highpass', can be a float that specifies pass frequency, or can be
        a tuple and is assumed to be (None, f_hi) for 'lowpass', and (f_lo, None) for 'highpass'.
    butterworth_order : int
        Order of the butterworth filter, if using an IIR filter.
        See input 'N' in scipy.signal.butter.
    plot_freq_response : bool, optional, default: False
        If True, plot the frequency response of the filter
    return_kernel : bool, optional, default: False
        If True, return the complex filter kernel.
    compute_transition_band : bool, optional, default: True
        If True, computes the transition bandwidth, and prints this information.

    Returns
    -------
    sig_filt : 1d array
        Filtered time series.
    kernel : length-2 tuple of arrays
        Filter kernel. Only returned if `return_kernel` is True.
    """

    # Design filter
    b_vals, a_vals = design_iir_filter(fs, pass_type, fc, butterworth_order)

    # Compute transition bandwidth
    if compute_transition_band:
        check_filter_properties(b_vals, a_vals, fs, pass_type, fc)

    # Remove any NaN on the edges of 'sig'
    sig, sig_nans = _remove_nans(sig)

    # Apply filter
    sig_filt = signal.filtfilt(b_vals, a_vals, sig)

    # Add NaN back on the edges of 'sig', if there were any at the beginning
    sig_filt = _restore_nans(sig_filt, sig_nans)

    # Plot frequency response, if desired
    if plot_freq_response:
        plot_frequency_response(fs, b_vals, a_vals)

    if return_kernel:
        return sig_filt, (b_vals, a_vals)
    else:
        return sig_filt


def design_fir_filter(sig_length, fs, pass_type, fc, n_cycles=3, n_seconds=None):
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
    fc : tuple of (float, float) or float
        Cutoff frequency(ies) used for filter, specified as f_lo & f_hi.
        For 'bandpass' & 'bandstop', must be a tuple.
        For 'lowpass' or 'highpass', can be a float that specifies pass frequency, or can be
        a tuple and is assumed to be (None, f_hi) for 'lowpass', and (f_lo, None) for 'highpass'.
    n_cycles : float, optional, default: 3
        Length of filter, in number of cycles, defined at the 'f_lo' frequency.
        This parameter is overwritten by `n_seconds`, if provided.
    n_seconds : float, optional
        Length of filter, in seconds.
        This parameter overwrites `n_cycles`.

    Returns
    -------
    kernel : 1d array
        The kernel definition of an FIR filter.
    """

    # Check filter definition
    f_lo, f_hi = check_filter_definition(pass_type, fc)
    filt_len = _fir_checks(pass_type, f_lo, f_hi, n_cycles, n_seconds, fs, sig_length)

    f_nyq = compute_nyquist(fs)
    if pass_type == 'bandpass':
        kernel = signal.firwin(filt_len, (f_lo, f_hi), pass_zero=False, nyq=f_nyq)
    elif pass_type == 'bandstop':
        kernel = signal.firwin(filt_len, (f_lo, f_hi), nyq=f_nyq)
    elif pass_type == 'highpass':
        kernel = signal.firwin(filt_len, f_lo, pass_zero=False, nyq=f_nyq)
    elif pass_type == 'lowpass':
        kernel = signal.firwin(filt_len, f_hi, nyq=f_nyq)

    return kernel


def design_iir_filter(fs, pass_type, fc, butterworth_order):
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
    fc : tuple of (float, float) or float
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
        B values for the filter.
    a_vals : 1d array
        A values for the filter.
    """

    # Warn about only recommending IIR for bandstop
    if pass_type != 'bandstop':
        warnings.warn('IIR filters are not recommended other than for notch filters.')

    # Check filter definition
    f_lo, f_hi = check_filter_definition(pass_type, fc)

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


def check_filter_definition(pass_type, fc):
    """Check a filter definition for validity, and get f_lo and f_hi.

    Parameters
    ----------
    pass_type : {'bandpass', 'bandstop', 'lowpass', 'highpass'}
        Which kind of filter to apply:

        * 'bandpass': apply a bandpass filter
        * 'bandstop': apply a bandstop (notch) filter
        * 'lowpass': apply a lowpass filter
        * 'highpass' : apply a highpass filter
    fc : tuple of (float, float) or float
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
        if isinstance(fc, tuple) and fc[0] >= fc[1]:
            raise ValueError('Second cutoff frequency must be greater than first.')
        elif isinstance(fc, (int, float)) or len(fc) != 2:
            raise ValueError('Two cutoff frequencies required for bandpass and bandstop filters')

        # Map fc to f_lo and f_hi
        f_lo, f_hi = fc

    # For lowpass and highpass can be tuple or int/float
    if pass_type == 'lowpass':
        if isinstance(fc, (int, float)):
            f_hi = fc
        elif isinstance(fc, tuple):
            f_hi = fc[1]
        f_lo = None

    if pass_type == 'highpass':
        if isinstance(fc, (int, float)):
            f_lo = fc
        elif isinstance(fc, tuple):
            f_lo = fc[0]
        f_hi = None

    # Make sure pass freqs are floats
    f_lo = float(f_lo) if f_lo else f_lo
    f_hi = float(f_hi) if f_hi else f_hi

    return f_lo, f_hi


def check_filter_properties(b_vals, a_vals, fs, pass_type, fc, transitions=(-20, -3)):
    """Check a filters properties, including pass band and transition band.

    Parameters
    ----------
    b_vals : 1d array
        B values for the filter.
    a_vals : 1d array
        A values for the filter.
    fs : float
        Sampling rate, in Hz.
    pass_type : {'bandpass', 'bandstop', 'lowpass', 'highpass'}
        Which kind of filter to apply:

        * 'bandpass': apply a bandpass filter
        * 'bandstop': apply a bandstop (notch) filter
        * 'lowpass': apply a lowpass filter
        * 'highpass' : apply a highpass filter
    fc : tuple of (float, float) or float
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
    pass_bw = compute_pass_band(fs, pass_type, fc)
    transition_bw = compute_trans_band(f_db, db, transitions[0], transitions[1])

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
        B values for the filter.
    a_vals : 1d array
        A values for the filter.
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


def compute_pass_band(fs, pass_type, fc):
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
    fc : tuple of (float, float) or float
        Cutoff frequency(ies) used for filter, specified as f_lo & f_hi.
        For 'bandpass' & 'bandstop', must be a tuple.
        For 'lowpass' or 'highpass', can be a float that specifies pass frequency, or can be
        a tuple and is assumed to be (None, f_hi) for 'lowpass', and (f_lo, None) for 'highpass'.

    Returns
    -------
    pass_bw : float
        The pass bandwidth of the filter.
    """

    f_lo, f_hi = check_filter_definition(pass_type, fc)
    if pass_type in ['bandpass', 'bandstop']:
        pass_bw = f_hi - f_lo
    elif pass_type == 'highpass':
        pass_bw = compute_nyquist(fs) - f_lo
    elif pass_type == 'lowpass':
        pass_bw = f_hi

    return pass_bw


def compute_trans_band(f_db, db, low=-20, high=-3):
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


def infer_passtype(fc):
    """Given frequency definition of a filter, infer the passtype.

    Parameters
    ----------
    fc : tuple of (float, float)
        Cutoff frequency(ies) used for filter, specified as f_lo & f_hi.

    Returns
    -------
    pass_type : str
        Which kind of filter pass_type is consistent with the frequency definition provided.

    Notes
    -----
    Assumes that a definition with two frequencies is a 'bandpass' (not 'bandstop').
    """

    if fc[0] is None:
        pass_type = 'lowpass'
    elif fc[1] is None:
        pass_type = 'highpass'
    else:
        pass_type = 'bandpass'

    # Check the inferred passtype & frequency definition is valid
    _ = check_filter_definition(pass_type, fc)

    return pass_type

###################################################################################################
###################################################################################################

def _remove_nans(sig):
    """Drop any NaNs on the edges of a 1d array.

    Parameters
    ----------
    sig : 1d array
        Signal to be checked for edge NaNs.

    Returns
    -------
    sig_removed : 1d array
        Signal with NaN edges removed.
    sig_nans : 1d array
        Boolean array indicating where NaNs were in the original array.
    """

    sig_nans = np.isnan(sig)
    sig_removed = sig[np.where(~np.isnan(sig))]

    return sig_removed, sig_nans


def _restore_nans(sig, sig_nans, dtype=float):
    """Restore NaN values to the edges of a 1d array.

    Parameters
    ----------
    sig : 1d array
        Signal that has had NaN edges removed.
    sig_nans : 1d array
        Boolean array indicating where NaNs were in the original array.

    Returns
    -------
    sig_restored : 1d array
        Signal with NaN edges restored.
    """

    sig_restored = np.ones(len(sig_nans), dtype=dtype) * np.nan
    sig_restored[~sig_nans] = sig

    return sig_restored


def _drop_edge_artifacts(sig, filt_len):
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


def _iir_checks(n_seconds, butterworth_order, remove_edge_artifacts):
    """Checks for using an IIR filter if called from the general filter function."""

    # Check inputs for IIR filters
    if n_seconds is not None:
        raise ValueError('n_seconds should not be defined for an IIR filter.')
    if butterworth_order is None:
        raise ValueError('butterworth_order must be defined when using an IIR filter.')
    if remove_edge_artifacts:
        warnings.warn('Edge artifacts are not removed when using an IIR filter.')
