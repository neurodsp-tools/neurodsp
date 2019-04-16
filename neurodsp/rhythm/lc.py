"""The lagged coherence algorithm for estimating the rhythmicity of a neural signal."""

import numpy as np

from scipy.signal.windows import hann
from neurodsp.utils.decorators import multidim

###################################################################################################
###################################################################################################

@multidim
def lagged_coherence(sig, f_range, fs, n_cycles=3, f_step=1, return_spectrum=False):
    """Quantify the rhythmicity of an oscillation using lagged coherence.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    f_range : list of float
        Frequency range of the oscillation, as [low, high], in Hz.
    fs : float
        Sampling rate, in Hz.
    n_cycles : float, optional, default: 3
        Number of cycles of the frequency of interest used to compute lagged coherence.
    f_step : float, optional, default: 1
        Step size to calculate lagged coherence in the frequency range, in Hz.
    return_spectrum : bool, optional, default: False
        If True, return the lagged coherence for all frequency values. Otherwise, only return mean.

    Returns
    -------
    lc : float or 1d array
        If return_spectrum is False: mean lagged coherence value in the frequency range of interest.
        If return_spectrum is True: lagged coherence value for each frequency across the range.
    freqs : 1d array
        Frequencies, in Hz, corresponding to the lagged coherence values in lc.
        Only returned if return_spectrum is True.

    References
    ----------
    Fransen, A. M., van Ede, F., & Maris, E. (2015).
    Identifying neuronal oscillations using rhythmicity.
    Neuroimage, 118, 256-267.
    """

    # Identify Fourier components of interest
    freqs = np.arange(f_range[0], f_range[1] + f_step, f_step)

    # Calculate lagged coherence for each frequency
    lc = np.zeros(len(freqs))
    for ind, freq in enumerate(freqs):
        lc[ind] = _lagged_coherence_1freq(sig, freq, fs, n_cycles=n_cycles)

    # Return desired measure of lagged coherence
    if return_spectrum:
        return lc, freqs
    else:
        return np.mean(lc)


def _lagged_coherence_1freq(sig, freq, fs, n_cycles=3):
    """Calculate lagged coherence of sig at frequency freq using the hanning-taper FFT method"""

    # Determine number of samples to be used in each window to compute lagged coherence
    n_samps = int(np.ceil(n_cycles * fs / freq))

    # For each N-cycle chunk, calculate the fourier coefficient at the frequency of interest, freq
    chunks = _nonoverlapping_chunks(sig, n_samps)
    chunks_len = len(chunks)

    hann_window = hann(n_samps)
    fourier_f = np.fft.fftfreq(n_samps, 1 / float(fs))
    fourier_f_idx = np.argmin(np.abs(fourier_f - freq))
    fourier_coefsoi = np.zeros(chunks_len, dtype=complex)

    for ind, chunk in enumerate(chunks):
        fourier_coef = np.fft.fft(chunk * hann_window)
        fourier_coefsoi[ind] = fourier_coef[fourier_f_idx]

    # Compute the lagged coherence value
    lcs_num = 0
    for ind in range(chunks_len - 1):
        lcs_num += fourier_coefsoi[ind] * np.conj(fourier_coefsoi[ind + 1])
    lcs_denom = np.sqrt(np.sum(np.abs(fourier_coefsoi[:-1])**2) * np.sum(np.abs(fourier_coefsoi[1:])**2))

    return np.abs(lcs_num / lcs_denom)


def _nonoverlapping_chunks(sig, n_samples):
    """Split sig into nonoverlapping chunks of length N"""

    n_chunks = int(np.floor(len(sig) / float(n_samples)))
    chunks = np.reshape(sig[:int(n_chunks * n_samples)], (n_chunks, int(n_samples)))

    return chunks
