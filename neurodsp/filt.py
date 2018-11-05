"""Filter a neural signal using a bandpass, highpass, lowpass, or bandstop filter."""

import warnings

import numpy as np
import scipy as sp
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
    # Check that frequency cutoff inputs are appropriate
    if pass_type in ('bandpass', 'bandstop'):
        # Check, if fc is a tuple and performing bandpass/stop, that
        #   the second cutoff frequency is greater than the first
        if isinstance(fc, tuple):
            if fc[0] >= fc[1]:
                raise ValueError('Second cutoff frequency must be greater than first.')

        if isinstance(fc, (int, float)):
            raise ValueError('Two cutoff frequencies required for bandpass and bandstop filters')
        if len(fc) != 2:
            raise ValueError('Two cutoff frequencies required for bandpass and bandstop filters')

        # Map fc to f_lo and f_hi
        f_lo, f_hi = fc

    # for LP and HP, a tuple or int/float can be passed
    #   if a tuple is passed, it's assumed (0,f_hi) for LP; (f_lo,Nyq) for HP
    # i.e., highpass is a bandpass with second cutoff freq at Nyquist freq
    if pass_type == 'lowpass':
        if isinstance(fc, (int, float)):
            f_hi = fc
        elif isinstance(fc, tuple):
            f_hi = fc[1]

    if pass_type == 'highpass':
        if isinstance(fc, (int, float)):
            f_lo = fc
        elif isinstance(fc, tuple):
            f_lo = fc[0]

    # Remove any NaN on the edges of 'sig'
    first_nonan = np.where(~np.isnan(sig))[0][0]
    last_nonan = np.where(~np.isnan(sig))[0][-1] + 1
    sig_old = np.copy(sig)
    sig = sig[first_nonan:last_nonan]

    # Process input for IIR filters
    if iir:
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

    # Process input for FIR filters
    else:
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

    # Compute nyquist frequency
    f_nyq = fs / 2.

    # Design filter
    if iir:
        if pass_type in ('bandpass', 'bandstop'):
            win = (f_lo / f_nyq, f_hi / f_nyq)
        elif pass_type == 'highpass':
            win = f_lo / f_nyq
        elif pass_type == 'lowpass':
            win = f_hi / f_nyq
        b_vals, a_vals = sp.signal.butter(butterworth_order, win, pass_type)
    else:
        if pass_type == 'bandpass':
            kernel = sp.signal.firwin(filt_len, (f_lo, f_hi), pass_zero=False, nyq=f_nyq)
        elif pass_type == 'bandstop':
            kernel = sp.signal.firwin(filt_len, (f_lo, f_hi), nyq=f_nyq)
        elif pass_type == 'highpass':
            kernel = sp.signal.firwin(filt_len, f_lo, pass_zero=False, nyq=f_nyq)
        elif pass_type == 'lowpass':
            kernel = sp.signal.firwin(filt_len, f_hi, nyq=f_nyq)

    # Apply filter
    if iir:
        sig_filt = sp.signal.filtfilt(b_vals, a_vals, sig)
    else:
        sig_filt = np.convolve(kernel, sig, 'same')

    # Plot frequency response, if desired
    if plot_freq_response:
        if iir:
            plot_frequency_response(fs, b_vals, a_vals)
        else:
            plot_frequency_response(fs, kernel)

    # Compute transition bandwidth
    if compute_transition_band and verbose:

        # Compute the frequency response in terms of Hz and dB
        if not iir:
            b_vals = kernel
            a_vals = 1

        w_vals, h_vals = signal.freqz(b_vals, a_vals)
        f_db = w_vals * fs / (2. * np.pi)
        db = 20 * np.log10(abs(h_vals))

        # Confirm frequency response goes below -20dB (significant attenuation)
        if np.min(db) >= -20:
            warnings.warn("The filter attenuation never goes below -20dB. "\
                          "Increase filter length.")
        else:
            # Compute pass bandwidth and transition bandwidth
            if pass_type == 'bandpass':

                if db[0] >= -20:
                    warnings.warn("The low frequency stopband never gets attenuated"\
                                  "by more than 20dB. Increase filter length.")
                if db[1] >= -20:
                    warnings.warn("The high frequency stopband never gets attenuated"\
                                  "by more than 20dB. Increase filter length.")

                pass_bw = f_hi - f_lo
                # Identify edges of transition band (-3dB and -20dB)
                cf_20db_1 = next(f_db[ind] for ind in range(len(db)) if db[ind] > -20)
                cf_3db_1 = next(f_db[ind] for ind in range(len(db)) if db[ind] > -3)
                cf_20db_2 = next(f_db[ind] for ind in range(len(db))[::-1] if db[ind] > -20)
                cf_3db_2 = next(f_db[ind] for ind in range(len(db))[::-1] if db[ind] > -3)
                # Compute transition bandwidth
                transition_bw1 = cf_3db_1 - cf_20db_1
                transition_bw2 = cf_20db_2 - cf_3db_2
                transition_bw = max(transition_bw1, transition_bw2)

            elif pass_type == 'bandstop':
                pass_bw = f_hi - f_lo
                # Identify edges of transition band (-3dB and -20dB)
                cf_20db_1 = next(f_db[ind] for ind in range(len(db)) if db[ind] < -20)
                cf_3db_1 = next(f_db[ind] for ind in range(len(db)) if db[ind] < -3)
                cf_20db_2 = next(f_db[ind] for ind in range(len(db))[::-1] if db[ind] < -20)
                cf_3db_2 = next(f_db[ind] for ind in range(len(db))[::-1] if db[ind] < -3)
                # Compute transition bandwidth
                transition_bw1 = cf_20db_1 - cf_3db_1
                transition_bw2 = cf_3db_2 - cf_20db_2
                transition_bw = max(transition_bw1, transition_bw2)

            elif pass_type == 'highpass':
                pass_bw = f_nyq - f_lo
                # Identify edges of transition band (-3dB and -20dB)
                cf_20db = next(f_db[ind] for ind in range(len(db)) if db[ind] > -20)
                cf_3db = next(f_db[ind] for ind in range(len(db)) if db[ind] > -3)
                # Compute transition bandwidth
                transition_bw = cf_3db - cf_20db

            elif pass_type == 'lowpass':
                pass_bw = f_hi
                # Identify edges of transition band (-3dB and -20dB)
                cf_20db = next(f_db[ind] for ind in range(len(db)) if db[ind] < -20)
                cf_3db = next(f_db[ind] for ind in range(len(db)) if db[ind] < -3)
                # Compute transition bandwidth
                transition_bw = cf_20db - cf_3db

            print('Transition bandwidth is ' + str(np.round(transition_bw, 1)) + \
                  ' Hz. Pass/stop bandwidth is ' + str(np.round(pass_bw, 1)) + ' Hz')

            # Raise warning if transition bandwidth is too high
            if transition_bw > pass_bw:
                warnings.warn('Transition bandwidth is ' + str(np.round(transition_bw, 1)) + \
                              ' Hz. This is greater than the desired pass/stop bandwidth of '\
                               + str(np.round(pass_bw, 1)) + ' Hz')

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
