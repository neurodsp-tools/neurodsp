"""Utility function for neurodsp.spectral."""

import numpy as np

###################################################################################################
###################################################################################################

def rotate_timeseries(sig, fs, delta_exp, f_rotation=1):
    """Rotate a timeseries of data, changing it's 1/f exponent.

    Parameters
    ----------
    sig : 1d array
        A time series to rotate.
    fs : float
        Sampling rate of the signal, in Hz.
    delta_exp : float
        Change in power law exponent to be applied.
        Positive is clockwise rotation (steepen), negative is counter clockwise rotation (flatten).
    f_rotation : float, optional, default: 1
        Frequency, in Hz, to rotate the spectrum around, where power is unchanged by the rotation.

    Returns
    -------
    sig_rotated : 1d array
        The rotated version of the signal.

    Notes
    -----
    This function works by taking the FFT and spectrally rotating the input signal.
    To return a timeseries, the rotated FFT is then turned back into a time series, with an iFFT.

    Examples
    --------
    Rotate a timeseries of simulated data:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> rotated_sig = rotate_timeseries(sig, fs=500, delta_exp=0.5)
    """

    # Compute the FFT
    fft_vals = np.fft.fft(sig)
    freqs = np.fft.fftfreq(len(sig), 1./fs)

    # Rotate the spectrum to create the exponent change
    #   Delta exponent is divided by two, as the FFT output is in units of amplitude not power
    fft_rotated = rotate_spectrum(freqs, fft_vals, delta_exp/2, f_rotation)

    # Invert back to time series, with a z-score to normalize
    sig_rotated = np.real(np.fft.ifft(fft_rotated))

    return sig_rotated


def rotate_spectrum(freqs, spectrum, delta_exponent, f_rotation=1):
    """Rotate the power law exponent of a power spectrum.

    Parameters
    ----------
    freqs : 1d array
        Frequency axis of input spectrum, in Hz.
    spectrum : 1d array
        Power spectrum to be rotated.
    delta_exponent : float
        Change in power law exponent to be applied.
        Positive is clockwise rotation (steepen), negative is counter clockwise rotation (flatten).
    f_rotation : float, optional, default: 1
        Frequency, in Hz, to rotate the spectrum around, where power is unchanged by the rotation.
        This only matters if not further normalizing signal variance.

    Returns
    -------
    rotated_spectrum : 1d array
        Rotated spectrum.

    Notes
    -----
    The input power spectrum is multiplied with a mask that applies the specified exponent change.

    Examples
    --------
    Rotate a power spectrum, calculated on simulated data:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.spectral import compute_spectrum
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs, spectrum = compute_spectrum(sig, fs=500)
    >>> rotated_spectrum = rotate_spectrum(freqs, spectrum, -1)
    """

    if freqs[0] == 0:
        skipped_zero = True
        f_0, p_0 = freqs[0], spectrum[0]
        freqs, spectrum = freqs[1:], spectrum[1:]
    else:
        skipped_zero = False

    mask = (np.abs(freqs) / f_rotation)**-delta_exponent
    rotated_spectrum = mask * spectrum

    if skipped_zero:
        freqs = np.insert(freqs, 0, f_0)
        rotated_spectrum = np.insert(rotated_spectrum, 0, p_0)

    return rotated_spectrum
