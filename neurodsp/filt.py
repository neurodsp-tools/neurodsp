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
                  verbose=True, compute_transition_band=True, remove_edge_artifacts=True):
    """Apply a bandpass, bandstop, highpass, or lowpass filter to a neural signal.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        The sampling rate, in Hz.
    pass_type : {'bandpass', 'bandstop', 'lowpass', 'highpass'}
        Which kind of filter to apply:

        * 'bandpass': apply a bandpass filter
        * 'bandstop': apply a bandstop (notch) filter
        * 'lowpass': apply a lowpass filter
        * 'highpass' : apply a highpass filter
    fc : tuple or float
        Cutoff frequency(ies) used for filter.
        Should be a tuple of 2 floats for bandpass and bandstop.
        Can be a tuple of 2 floats, or a single float, for low/highpass.
        If float, it's taken as the cutoff frequency. If tuple, it's assumed
        as (None, f_hi) for LP, and (f_lo, None) for HP.
    n_cycles : float, optional, default: 3
        Length of filter in terms of number of cycles at 'f_lo' frequency.
        This parameter is overwritten by 'n_seconds', if provided.
    n_seconds : float, optional
        Length of filter, in seconds.
    iir : bool, optional
        If True, use an infinite-impulse response (IIR) filter.
        The only IIR filter to be used is a butterworth filter.
    butterworth_order : int, optional
        Order of the butterworth filter.
        See input 'N' in scipy.signal.butter.
    plot_freq_response : bool, optional
        If True, plot the frequency response of the filter
    return_kernel : bool, optional
        If True, return the complex filter kernel
    verbose : bool, optional
        If True, print optional information
    compute_transition_band : bool, optional
        If True, the function computes the transition bandwidth,
        defined as the frequency range between -20dB and -3dB attenuation,
        and warns the user if this band is longer than the frequency bandwidth.
    remove_edge_artifacts : bool, optional
        If True, replace the samples that are within half a kernel's length to
        the signal edge with np.nan.

    Returns
    -------
    sig_filt : 1d array
        Filtered time series.
    kernel : length-2 tuple of arrays
        Filter kernel. Only returned if 'return_kernel' is True.
    """

    # Check inputs & compute the nyquist frequency
    f_lo, f_hi = check_filter_definitions(pass_type, fc)

    # Remove any NaN on the edges of 'sig'
    sig, sig_nans = remove_nans(sig)

    if iir:
        iir_checks(pass_type, n_seconds, butterworth_order, remove_edge_artifacts, verbose)
        sig_filt, b_vals, a_vals = iir_filt(sig, fs, pass_type, f_lo, f_hi,
                                            butterworth_order, plot_freq_response)
    else:
        sig_filt, filt_len, kernel = fir_filt(sig, fs, pass_type, f_lo, f_hi, n_cycles,
                                              n_seconds, plot_freq_response)

    # Compute transition bandwidth
    if compute_transition_band and verbose:
        if not iir:
            a_vals, b_vals = 1, kernel
        calc_transition_band(b_vals, a_vals, fs, f_lo, f_hi, pass_type)

    # Remove edge artifacts
    if not iir and remove_edge_artifacts:
        sig_filt = drop_edge_artifacts(sig_filt, filt_len)

    # Add NaN back on the edges of 'sig', if there were any at the beginning
    sig_filt = restore_nans(sig_filt, sig_nans)

    # Return kernel if desired
    if return_kernel:
        if iir:
            return sig_filt, (b_vals, a_vals)
        else:
            return sig_filt, kernel
    else:
        return sig_filt


def remove_nans(sig):
    """Words, words, words."""

    sig_nans = np.isnan(sig)
    sig_removed = sig[np.where(~np.isnan(sig))]

    return sig_removed, sig_nans

def restore_nans(sig, sig_nans):
    """Words, words, words."""

    sig_restored = np.ones(len(sig_nans)) * np.nan
    sig_restored[~sig_nans] = sig

    return sig_restored


def calculate_nyquist(fs):
    """Calculate the nyquist frequency."""

    return fs / 2.


def drop_edge_artifacts(sig, filt_len):
    """Words, words, words."""

    n_rmv = int(np.ceil(filt_len / 2))
    sig[:n_rmv] = np.nan
    sig[-n_rmv:] = np.nan

    return sig


def check_filter_definitions(pass_type, fc):
    """Words, words, words."""

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

    # For LP and HP can be tuple or int/float
    #   Tuple is assumed to be (0, f_hi) for LP; (f_lo, f_nyq) for HP
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

    return f_lo, f_hi


def iir_checks(pass_type, n_seconds, butterworth_order, remove_edge_artifacts, verbose):
    """Checks for using an IIR filter if called from the general filter function."""

    # Check input for IIR filters
    if remove_edge_artifacts:
        if verbose:
            warnings.warn('Edge artifacts are not removed when using an IIR filter.')
    if pass_type != 'bandstop':
        if verbose:
            warnings.warn('IIR filters are not recommended other than for notch filters.')
    if n_seconds is not None:
        raise TypeError('n_seconds should not be defined for an IIR filter.')
    if butterworth_order is None:
        raise TypeError('butterworth_order must be defined when using an IIR filter.')


def iir_filt(sig, fs, pass_type, f_lo, f_hi, butterworth_order, plot_freq_response):
    """Words, words, words."""

    f_nyq = calculate_nyquist(fs)

    if pass_type in ('bandpass', 'bandstop'):
        win = (f_lo / f_nyq, f_hi / f_nyq)
    elif pass_type == 'highpass':
        win = f_lo / f_nyq
    elif pass_type == 'lowpass':
        win = f_hi / f_nyq

    # Design & apply filter
    b_vals, a_vals = signal.butter(butterworth_order, win, pass_type)
    sig_filt = signal.filtfilt(b_vals, a_vals, sig)

    # Plot frequency response, if desired
    if plot_freq_response:
        plot_frequency_response(fs, b_vals, a_vals)

    return sig_filt, b_vals, a_vals


def fir_filt(sig, fs, pass_type, f_lo, f_hi, n_cycles, n_seconds, plot_freq_response):
    """Words, words, words."""

    # Compute filter length if specified in seconds
    if n_seconds is not None:
        filt_len = int(np.ceil(fs * n_seconds))
    else:
        if pass_type == 'lowpass':
            filt_len = int(np.ceil(fs * n_cycles / f_hi))
        else:
            filt_len = int(np.ceil(fs * n_cycles / f_lo))
    f_nyq = calculate_nyquist(fs)

    # Force filter length to be odd
    if filt_len % 2 == 0:
        filt_len = int(filt_len + 1)

    # Raise an error if the filter is longer than the signal
    if filt_len >= len(sig):
        raise ValueError(
            """The designed filter (length: {:d}) is longer than the signal (length: {:d}).
            The filter needs to be shortened by decreasing the n_cycles or n_seconds parameter.
            However, this will decrease the frequency resolution of the filter.""".format(filt_len, len(sig)))

    # Design filter
    if pass_type == 'bandpass':
        kernel = signal.firwin(filt_len, (f_lo, f_hi), pass_zero=False, nyq=f_nyq)
    elif pass_type == 'bandstop':
        kernel = signal.firwin(filt_len, (f_lo, f_hi), nyq=f_nyq)
    elif pass_type == 'highpass':
        kernel = signal.firwin(filt_len, f_lo, pass_zero=False, nyq=f_nyq)
    elif pass_type == 'lowpass':
        kernel = signal.firwin(filt_len, f_hi, nyq=f_nyq)

    # Apply filter
    sig_filt = np.convolve(kernel, sig, 'same')

    # Plot frequency response, if desired
    if plot_freq_response:
        plot_frequency_response(fs, kernel)

    return sig_filt, filt_len, kernel


def calc_transition_band(b_vals, a_vals, fs, f_lo, f_hi, pass_type, transitions=(-20, -3)):
    """Words, words, words."""

    # Compute the frequency response in terms of Hz and dB
    w_vals, h_vals = signal.freqz(b_vals, a_vals)
    f_db = w_vals * fs / (2. * np.pi)
    db = 20 * np.log10(abs(h_vals))

    # Check that frequency response goes below transition level (has significant attenuation)
    if np.min(db) >= transitions[0]:
        warnings.warn("The filter attenuation never goes below -20dB. "\
                      "Increase filter length.".format(transitions[0]))
        # If there is no attenuation, cannot calculate bands - so return here
        return

    # Check that both sides of a bandpass have significant attenuation
    if pass_type == 'bandpass':
        if db[0] >= transitions[0] or db[-1] >= transitions[0]:
            warnings.warn("The low or high frequency stopband never gets attenuated by"\
                          "more than {} dB. Increase filter length.".format(abs(transitions[0])))

    # Compute pass bandwidth
    if pass_type in ['bandpass', 'bandstop']:
        pass_bw = f_hi - f_lo
    elif pass_type == 'highpass':
        pass_bw = calculate_nyquist(fs) - f_lo
    elif pass_type == 'lowpass':
        pass_bw = f_hi

    # Compute transition bandwidth
    transition_bw = get_transition_band(db, f_db, transitions[0], transitions[1])

    # Raise warning if transition bandwidth is too high
    if transition_bw > pass_bw:
        warnings.warn('Transition bandwidth is  {:.1f}  Hz. This is greater than the desired'\
                      'pass/stop bandwidth of  {:.1f} Hz'.format(transition_bw, pass_bw))

    # Print out things
    print('Transition bandwidth is {:.1f} Hz.'.format(transition_bw))
    print('Pass/stop bandwidth is {:.1f} Hz'.format(pass_bw))


def get_transition_band(db, f_db, low, high):
    """Words, words, words."""

    # This gets the indices of transitions to the values in searched for range
    inds = np.where(np.diff(np.logical_and(db > low, db < high)))[0]
    # This steps through the indices, in pairs, selecting from the vector to select from
    trans_band = np.max([(b - a) for a, b in zip(f_db[inds[0::2]], f_db[inds[1::2]])])

    return trans_band
