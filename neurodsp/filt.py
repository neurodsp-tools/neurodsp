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
    sig : array-like 1d
        Voltage time series
    fs : float
        The sampling rate
    pass_type : str
        'bandpass' : apply a bandpass filter
        'bandstop' : apply a bandstop (notch) filter
        'lowpass' : apply a lowpass filter
        'highpass' : apply a highpass filter
    fc : tuple or float
        cutoff frequency(ies) used for filter
        Should be a tuple of 2 floats for bandpass and bandstop
        Can be a tuple of 2 floats, or a single float, for low/highpass.
        If float, it's taken as the cutoff frequency. If tuple, it's assumed
            as (None,f_hi) for LP, and (f_lo,None) for HP.
    n_cycles : float, optional
        Length of filter in terms of number of cycles at 'f_lo' frequency
        This parameter is overwritten by 'n_seconds'
    n_seconds : float, optional
        Length of filter (seconds)
    iir : bool, optional
        if True, use an infinite-impulse response (IIR) filter
        The only IIR filter to be used is a butterworth filter
    butterworth_order : int, optional
        order of the butterworth filter
        see input 'N' in scipy.signal.butter
    plot_freq_response : bool, optional
        if True, plot the frequency response of the filter
    return_kernel : bool, optional
        if True, return the complex filter kernel
    verbose : bool, optional
        if True, print optional information
    compute_transition_band : bool, optional
        if True, the function computes the transition bandwidth,
        defined as the frequency range between -20dB and -3dB attenuation,
        and warns the user if this band is longer than the frequency bandwidth.
    remove_edge_artifacts : bool, optional
        if True, replace the samples that are within half a kernel's length to
        the signal edge with np.nan

    Returns
    -------
    sig_filt : array-like 1d
        filtered time series
    kernel : length-2 tuple of arrays
        filter kernel
        returned only if 'return_kernel' == True
    """

    # Check inputs &  compute the nyquist frequency
    f_lo, f_hi = check_filter_definitions(pass_type, fc)
    f_nyq = fs / 2.

    # Remove any NaN on the edges of 'sig'
    first_nonan = np.where(~np.isnan(sig))[0][0]
    last_nonan = np.where(~np.isnan(sig))[0][-1] + 1
    sig_old = np.copy(sig)
    sig = sig[first_nonan:last_nonan]

    if iir:
        iir_checks(pass_type, n_seconds, butterworth_order, remove_edge_artifacts, verbose)
        sig_filt, b_vals, a_vals = iir_filt(sig, pass_type, f_lo, f_hi, f_nyq,
                                            butterworth_order, plot_freq_response)
    else:
        sig_filt, filt_len, kernel = fir_filt(sig, fs, pass_type, f_lo, f_hi, f_nyq, n_cycles,
                                              n_seconds, plot_freq_response)

    # Compute transition bandwidth
    if compute_transition_band and verbose:
        if not iir:
            a_vals, b_vals = 1, kernel
        calc_transition_band(b_vals, a_vals, fs, f_lo, f_hi, f_nyq, pass_type)

    # Remove edge artifacts
    if not iir and remove_edge_artifacts:
        n_rmv = int(np.ceil(filt_len / 2))
        sig_filt[:n_rmv] = np.nan
        sig_filt[-n_rmv:] = np.nan

    # Add NaN back on the edges of 'x', if there were any at the beginning
    sig_filt_full = np.ones(len(sig_old)) * np.nan
    sig_filt_full[first_nonan:last_nonan] = sig_filt
    sig_filt = sig_filt_full

    # Return kernel if desired
    if return_kernel:
        if iir:
            return sig_filt, (b_vals, a_vals)
        else:
            return sig_filt, kernel
    else:
        return sig_filt


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
    #   Tuple is assumed to be (0, f_hi) for LP; (f_lo, Nyq) for HP
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


def iir_filt(sig, pass_type, f_lo, f_hi, f_nyq, butterworth_order, plot_freq_response):
    """Words, words, words."""

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


def fir_filt(sig, fs, pass_type, f_lo, f_hi, f_nyq, n_cycles, n_seconds, plot_freq_response):
    """Words, words, words."""

    # Compute filter length if specified in seconds
    if n_seconds is not None:
        filt_len = int(np.ceil(fs * n_seconds))
    else:
        if pass_type == 'lowpass':
            filt_len = int(np.ceil(fs * n_cycles / f_hi))
        else:
            filt_len = int(np.ceil(fs * n_cycles / f_lo))

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


def calc_transition_band(b_vals, a_vals, fs, f_lo, f_hi, f_nyq, pass_type):
    """Words, words, words."""

    # Compute the frequency response in terms of Hz and dB
    w_vals, h_vals = signal.freqz(b_vals, a_vals)
    f_db = w_vals * fs / (2. * np.pi)
    db = 20 * np.log10(abs(h_vals))

    # Confirm frequency response goes below -20dB (significant attenuation)
    if np.min(db) >= -20:
        warnings.warn("The filter attenuation never goes below -20dB. "\
                      "Increase filter length.")
    else:
        if pass_type == 'bandpass':

            if db[0] >= -20:
                warnings.warn("The low frequency stopband never gets attenuated"\
                              "by more than 20dB. Increase filter length.")
            if db[1] >= -20:
                warnings.warn("The high frequency stopband never gets attenuated"\
                              "by more than 20dB. Increase filter length.")

        # Compute pass bandwidth and transition bandwidth
        if pass_type in ['bandpass', 'bandstop']:
            pass_bw = f_hi - f_lo
        elif pass_type == 'highpass':
            pass_bw = f_nyq - f_lo
        elif pass_type == 'lowpass':
            pass_bw = f_hi

        transition_bw = get_transition_band(db, f_db, -20, -3)

        print('Transition bandwidth is ' + str(np.round(transition_bw, 1)) + \
              ' Hz. Pass/stop bandwidth is ' + str(np.round(pass_bw, 1)) + ' Hz')

        # Raise warning if transition bandwidth is too high
        if transition_bw > pass_bw:
            warnings.warn('Transition bandwidth is ' + str(np.round(transition_bw, 1)) + \
                          ' Hz. This is greater than the desired pass/stop bandwidth of '\
                           + str(np.round(pass_bw, 1)) + ' Hz')


def get_transition_band(vec, sel, low, high):
    """Words, words, words."""

    # This gets the indices of transitions to the values in searched for range
    inds = np.where(np.diff(np.logical_and(vec > low, vec < high)))[0]
    # This steps through the indices, in pairs, selecting from the vector to select from
    trans_band = np.max([(b - a) for a, b in zip(sel[inds[0::2]], sel[inds[1::2]])])

    return trans_band
