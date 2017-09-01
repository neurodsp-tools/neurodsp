"""
filt.py
Filter a neural signal using a bandpass, highpass, lowpass, or bandstop filter.
"""

import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
import warnings


def filter(x, Fs, pass_type, f_lo=None, f_hi=None, N_cycles=3, N_seconds=None,
           iir=False, butterworth_order=None,
           plot_frequency_response=False, return_kernel=False,
           verbose=True, compute_transition_band=True, remove_edge_artifacts=True):
    """
    Apply a bandpass, bandstop, highpass, or lowpass filter to a neural signal

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Fs : float
        The sampling rate
    pass_type : str
        'bandpass' : apply a bandpass filter
        'bandstop' : apply a bandstop (notch) filter
        'lowpass' : apply a lowpass filter
        'highpass' : apply a highpass filter
    f_lo : float
        Low-frequency cutoff (Hz)
    f_hi : float
        High-frequency cutoff (Hz)
    N_cycles : float
        Length of filter in terms of number of cycles at 'f_lo' frequency
        This parameter is overwritten by 'N_seconds'
    N_seconds : float
        Length of filter (seconds)
    iir : bool
        if True, use an infinite-impulse response (IIR) filter
        The only IIR filter to be used is a butterworth filter
    butterworth_order : int
        order of the butterworth filter
        see input 'N' in scipy.signal.butter
    plot_frequency_response : bool
        if True, plot the frequency response of the filter
    return_kernel : bool
        if True, return the complex filter kernel
    verbose : bool
        if True, print optional information
    compute_transition_band : bool
        if True, the function computes the transition bandwidth,
        defined as the frequency range between -20dB and -3dB attenuation,
        and warns the user if this band is longer than the frequency bandwidth.
    remove_edge_artifacts : bool
        if True, replace the samples that are within half a kernel's length to
        the signal edge with np.nan

    Returns
    -------
    x_filt : array-like 1d
        filtered time series
    kernel : length-2 tuple of arrays
        filter kernel
        returned only if 'return_kernel' == True
    """

    # Check that frequency cutoff inputs are appropriate
    if pass_type == 'bandpass' or pass_type == 'bandstop':
        if f_lo is None or f_hi is None:
            raise ValueError('For bandpass and bandstop filters, both f_lo and f_hi must be defined.')
        if f_lo >= f_hi:
            raise ValueError('f_lo must be less than f_hi.')

    elif pass_type == 'highpass':
        if f_lo is not None:
            raise ValueError('For highpass filters, f_lo must not be used. f_hi is used to indicate the cutoff frequency.')
        if f_hi is None:
            raise ValueError('For highpass filters, f_hi must be used to represent the cutoff frequency. f_lo is not used')

    elif pass_type == 'lowpass':
        if f_hi is not None:
            raise ValueError('For lowpass filters, f_hi must not be used. f_lo is used to indicate the cutoff frequency.')
        if f_lo is None:
            raise ValueError('For lowpass filters, f_lo must be used to represent the cutoff frequency. f_hi is not used')

    else:
        raise ValueError('Input for "pass_type" not recognized. Must indicate bandpass, bandstop, lowpass, or highpass.')

    # Remove any NaN on the edges of 'x'
    first_nonan = np.where(~np.isnan(x))[0][0]
    last_nonan = np.where(~np.isnan(x))[0][-1] + 1
    x_old = np.copy(x)
    x = x[first_nonan:last_nonan]

    # Process input for IIR filters
    if iir:
        if remove_edge_artifacts:
            if verbose:
                warnings.warn('Edge artifacts are not removed when using an IIR filter.')
        if pass_type != 'bandstop':
            if verbose:
                warnings.warn('IIR filters are not recommended other than for notch filters.')
        if N_seconds is not None:
            raise TypeError('N_seconds should not be defined for an IIR filter.')
        if butterworth_order is None:
            raise TypeError('butterworth_order must be defined when using an IIR filter.')

    # Process input for FIR filters
    else:
        # Compute filter length if specified in seconds
        if N_seconds is not None:
            N = int(np.ceil(Fs * N_seconds))
        else:
            if pass_type == 'highpass':
                N = int(np.ceil(Fs * N_cycles / f_hi))
            else:
                N = int(np.ceil(Fs * N_cycles / f_lo))

        # Force filter length to be odd
        if N % 2 == 0:
            N = int(N + 1)

    # Compute nyquist frequency
    f_nyq = Fs / 2.

    # Design filter
    if iir:
        if pass_type == 'bandpass' or pass_type == 'bandstop':
            Wn = (f_lo / f_nyq, f_hi / f_nyq)
        elif pass_type == 'highpass':
            Wn = f_hi / f_nyq
        elif pass_type == 'lowpass':
            Wn = f_lo / f_nyq
        b, a = sp.signal.butter(butterworth_order, Wn, pass_type)
    else:
        if pass_type == 'bandpass':
            kernel = sp.signal.firwin(N, (f_lo, f_hi), pass_zero=False, nyq=f_nyq)
        elif pass_type == 'bandstop':
            kernel = sp.signal.firwin(N, (f_lo, f_hi), nyq=f_nyq)
        elif pass_type == 'highpass':
            kernel = sp.signal.firwin(N, f_hi, pass_zero=False, nyq=f_nyq)
        elif pass_type == 'lowpass':
            kernel = sp.signal.firwin(N, f_lo, nyq=f_nyq)

    # Apply filter
    if iir:
        x_filt = sp.signal.filtfilt(b, a, x)
    else:
        x_filt = np.convolve(kernel, x, 'same')

    # Plot frequency response, if desired
    if plot_frequency_response:
        if iir:
            _plot_frequency_response(Fs, b, a)
        else:
            _plot_frequency_response(Fs, kernel)

    # Compute transition bandwidth
    if compute_transition_band and verbose:

        # Compute the frequency response in terms of Hz and dB
        if not iir:
            b = kernel
            a = 1
        w, h = signal.freqz(b, a)
        f_db = w * Fs / (2. * np.pi)
        db = 20 * np.log10(abs(h))

        # Compute pass bandwidth and transition bandwidth
        if pass_type == 'bandpass':
            pass_bw = f_hi - f_lo
            # Identify edges of transition band (-3dB and -20dB)
            cf_20db_1 = next(f_db[i] for i in range(len(db)) if db[i] > -20)
            cf_3db_1 = next(f_db[i] for i in range(len(db)) if db[i] > -3)
            cf_20db_2 = next(f_db[i] for i in range(len(db))[::-1] if db[i] > -20)
            cf_3db_2 = next(f_db[i] for i in range(len(db))[::-1] if db[i] > -3)
            # Compute transition bandwidth
            transition_bw1 = cf_3db_1 - cf_20db_1
            transition_bw2 = cf_20db_2 - cf_3db_2
            transition_bw = max(transition_bw1, transition_bw2)

            if cf_20db_1 == f_db[0]:
                warnings.warn('The low frequency stopband never gets attenuated by more than 20dB. Increase filter length.')
            if cf_20db_2 == f_db[-1]:
                warnings.warn('The high frequency stopband never gets attenuated by more than 20dB. Increase filter length.')

        elif pass_type == 'bandstop':
            pass_bw = f_hi - f_lo
            # Identify edges of transition band (-3dB and -20dB)
            cf_20db_1 = next(f_db[i] for i in range(len(db)) if db[i] < -20)
            cf_3db_1 = next(f_db[i] for i in range(len(db)) if db[i] < -3)
            cf_20db_2 = next(f_db[i] for i in range(len(db))[::-1] if db[i] < -20)
            cf_3db_2 = next(f_db[i] for i in range(len(db))[::-1] if db[i] < -3)
            # Compute transition bandwidth
            transition_bw1 = cf_20db_1 - cf_3db_1
            transition_bw2 = cf_3db_2 - cf_20db_2
            transition_bw = max(transition_bw1, transition_bw2)

        elif pass_type == 'highpass':
            pass_bw = f_hi
            # Identify edges of transition band (-3dB and -20dB)
            cf_20db = next(f_db[i] for i in range(len(db)) if db[i] > -20)
            cf_3db = next(f_db[i] for i in range(len(db)) if db[i] > -3)
            # Compute transition bandwidth
            transition_bw = cf_3db - cf_20db

        elif pass_type == 'lowpass':
            pass_bw = f_lo
            # Identify edges of transition band (-3dB and -20dB)
            cf_20db = next(f_db[i] for i in range(len(db)) if db[i] < -20)
            cf_3db = next(f_db[i] for i in range(len(db)) if db[i] < -3)
            # Compute transition bandwidth
            transition_bw = cf_20db - cf_3db

        # Raise warning if transition bandwidth is too high
        if transition_bw > pass_bw:
            warnings.warn('Transition bandwidth is ' + str(np.round(transition_bw, 1)) + ' Hz. This is greater than the desired pass/stop bandwidth of ' + str(np.round(pass_bw, 1)) + ' Hz')

    # Remove edge artifacts
    if not iir and remove_edge_artifacts:
        N_rmv = int(np.ceil(N / 2))
        x_filt[:N_rmv] = np.nan
        x_filt[-N_rmv:] = np.nan

    # Add NaN back on the edges of 'x', if there were any at the beginning
    x_filt_full = np.ones(len(x_old)) * np.nan
    x_filt_full[first_nonan:last_nonan] = x_filt
    x_filt = x_filt_full

    # Return kernel if desired
    if return_kernel:
        if iir:
            return x_filt, (b, a)
        else:
            return x_filt, kernel
    else:
        return x_filt


def _plot_frequency_response(Fs, b, a=1):
    """Compute frequency response of a filter kernel b with sampling rate Fs"""
    w, h = signal.freqz(b, a)
    # Plot frequency response
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(w * Fs / (2. * np.pi), 20 * np.log10(abs(h)), 'k')
    plt.title('Frequency response')
    plt.ylabel('Attenuation (dB)')
    plt.xlabel('Frequency (Hz)')
    if type(a) is int:
        # Plot filter kernel
        plt.subplot(1, 2, 2)
        plt.plot(b, 'k')
        plt.title('Kernel')
    plt.show()
