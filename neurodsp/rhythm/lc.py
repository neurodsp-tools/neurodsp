"""The lagged coherence algorithm for estimating the rhythmicity of a neural signal."""

from warnings import warn

import numpy as np

from scipy.signal.windows import hann

from neurodsp.utils.data import create_freqs
from neurodsp.utils.decorators import multidim

###################################################################################################
###################################################################################################

@multidim
def lagged_coherence(sig, fs, freqs, n_cycles=3, return_spectrum=False):
    """Quantify the rhythmicity of a frequency range using lagged coherence.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    freqs : 1d array or list of float
        If array, frequency values to estimate with morlet wavelets.
        If list, define the frequency range, as [freq_start, freq_stop, freq_step].
        The `freq_step` is optional, and defaults to 1. Range is inclusive of `freq_stop` value.
    n_cycles : float, optional, default: 3
        Number of cycles of each frequency to use to compute lagged coherence.
    return_spectrum : bool, optional, default: False
        If True, return the lagged coherence for all frequency values.
        Otherwise, only the mean lagged coherence value across the frequency range is returned.

    Returns
    -------
    lc : float or 1d array
        If return_spectrum is False: mean lagged coherence value in the frequency range of interest.
        If return_spectrum is True: lagged coherence value for each frequency across the range.
    freqs : 1d array
        Frequencies, corresponding to the lagged coherence values, in Hz.
        Only returned if return_spectrum is True.

    References
    ----------
    Fransen, A. M., van Ede, F., & Maris, E. (2015).
    Identifying neuronal oscillations using rhythmicity.
    Neuroimage, 118, 256-267.
    """

    if isinstance(freqs, (tuple, list)):
        freqs = create_freqs(*freqs)

    # Calculate lagged coherence for each frequency
    lc = np.zeros(len(freqs))
    for ind, freq in enumerate(freqs):
        lc[ind] = _lagged_coherence_1freq(sig, freq, fs, n_cycles=n_cycles)

    # Check if all values were properly estimated
    if sum(np.isnan(lc)) > 0:
        warn("NEURODSP - LAGGED COHERENCE WARNING:"
             "\nLagged coherence could not be estimated for at least some requested frequencies."
             "\nThis happens, especially with low frequencies, when there are not enough samples "
             "per segment and/or not enough segments available to estimate the measure."
             "\nTry using a greater number of cycles and/or a longer signal length, and/or adjust the frequency range.")

    # Return desired measure of lagged coherence
    if return_spectrum:
        return lc, freqs
    else:
        return np.mean(lc)


def _lagged_coherence_1freq(sig, freq, fs, n_cycles=3):
    """Calculate lagged coherence of a specific frequency using the hanning-taper FFT method"""

    # Determine number of samples to be used in each window to compute lagged coherence
    n_samps = int(np.ceil(n_cycles * fs / freq))

    # Split the signal into chunks
    chunks = _nonoverlapping_chunks(sig, n_samps)
    n_chunks = len(chunks)

    # For each chunk, calculate the fourier coefficients at the frequency of interest
    hann_window = hann(n_samps)
    fft_freqs = np.fft.fftfreq(n_samps, 1 / float(fs))
    fft_freqs_idx = np.argmin(np.abs(fft_freqs - freq))

    fft_coefs = np.zeros(n_chunks, dtype=complex)
    for ind, chunk in enumerate(chunks):
        fourier_coef = np.fft.fft(chunk * hann_window)
        fft_coefs[ind] = fourier_coef[fft_freqs_idx]

    # Compute the lagged coherence value
    lcs_num = 0
    for ind in range(n_chunks - 1):
        lcs_num += fft_coefs[ind] * np.conj(fft_coefs[ind + 1])
    lcs_denom = np.sqrt(np.sum(np.abs(fft_coefs[:-1])**2) * np.sum(np.abs(fft_coefs[1:])**2))

    return np.abs(lcs_num / lcs_denom)


def _nonoverlapping_chunks(sig, n_samples):
    """Split a signal into non-overlapping chunks."""

    n_chunks = int(np.floor(len(sig) / float(n_samples)))
    chunks = np.reshape(sig[:int(n_chunks * n_samples)], (n_chunks, int(n_samples)))

    return chunks
