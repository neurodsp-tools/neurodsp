"""The lagged coherence algorithm for estimating the rhythmicity of a neural signal."""

from warnings import warn

import numpy as np
from scipy.signal.windows import hann

from neurodsp.utils.checks import check_n_cycles
from neurodsp.utils.data import create_freqs, split_signal
from neurodsp.utils.decorators import multidim

###################################################################################################
###################################################################################################

@multidim(select=[1])
def compute_lagged_coherence(sig, fs, freqs, n_cycles=3, return_spectrum=False):
    """Compute lagged coherence, reflecting the rhythmicity across a frequency range.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    freqs : 1d array or list of float
        The frequency values at which to estimate lagged coherence.
        If array, defines the frequency values to use.
        If list, define the frequency range, as [freq_start, freq_stop, freq_step].
        The `freq_step` is optional, and defaults to 1. Range is inclusive of `freq_stop` value.
    n_cycles : float or list of float, default: 3
        The number of cycles to use to compute lagged coherence, for each frequency.
        If a single value, the same number of cycles is used for each frequency.
        If a list or list_like, there should be a value corresponding to each frequency.
    return_spectrum : bool, optional, default: False
        If True, return the lagged coherence for all frequency values.
        Otherwise, only the mean lagged coherence value across the frequency range is returned.

    Returns
    -------
    lcs : float or 1d array
        If `return_spectrum` is False: mean lagged coherence value across the frequency range.
        If `return_spectrum` is True: lagged coherence values for all frequencies.
    freqs : 1d array
        Frequencies, corresponding to the lagged coherence values, in Hz.
        Only returned if `return_spectrum` is True.

    Notes
    -----
    - The lagged coherence algorithm is described in [1]_.

    References
    ----------
    .. [1] Fransen, A. M., van Ede, F., & Maris, E. (2015). Identifying neuronal
           oscillations using rhythmicity. NeuroImage, 118, 256-267.
           DOI: https://doi.org/10.1016/j.neuroimage.2015.06.003

    Examples
    --------
    Compute lagged coherence for a simulated signal with beta oscillations:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_synaptic_current': {},
    ...                                'sim_bursty_oscillation': {'freq': 20,
    ...                                                           'enter_burst': .50,
    ...                                                           'leave_burst': .25}})
    >>> lag_cohs = compute_lagged_coherence(sig, fs=500, freqs=(5, 35))
    """

    if isinstance(freqs, (tuple, list)):
        freqs = create_freqs(*freqs)
    n_cycles = check_n_cycles(n_cycles, len(freqs))

    # Calculate lagged coherence for each frequency
    lcs = np.zeros(len(freqs))
    for ind, (freq, n_cycle) in enumerate(zip(freqs, n_cycles)):
        lcs[ind] = lagged_coherence_1freq(sig, fs, freq, n_cycles=n_cycle)

    # Check if all values were properly estimated
    if sum(np.isnan(lcs)) > 0:
        warn("NEURODSP - LAGGED COHERENCE WARNING:"
             "\nLagged coherence could not be estimated for at least some requested frequencies."
             "\nThis happens, especially with low frequencies, when there are not enough samples "
             "per segment and/or not enough segments available to estimate the measure."
             "\nTry using a greater number of cycles and/or a longer signal length, and/or "
             "adjust the frequency range.")

    if return_spectrum:
        return lcs, freqs
    else:
        return np.mean(lcs)


def lagged_coherence_1freq(sig, fs, freq, n_cycles):
    """Compute the lagged coherence at a particular frequency.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    freq : float
        The frequency at which to estimate lagged coherence.
    n_cycles : float
        The number of cycles of the given frequency to use to compute lagged coherence.

    Returns
    -------
    lc : float
        The computed lagged coherence value.

    Notes
    -----
    - Lagged coherence is computed using hanning-tapered FFTs.
    - The returned lagged coherence value is bound between 0 and 1.
    """

    # Determine number of samples to be used in each window to compute lagged coherence
    n_samps = int(np.ceil(n_cycles * fs / freq))

    # Split the signal into chunks
    chunks = split_signal(sig, n_samps)
    n_chunks = len(chunks)

    # Create the window to apply to each chunk
    hann_window = hann(n_samps)

    # Create the frequency vector, finding the frequency value of interest
    fft_freqs = np.fft.fftfreq(n_samps, 1 / float(fs))
    fft_freqs_idx = np.argmin(np.abs(fft_freqs - freq))

    # Calculate the Fourier coefficients across chunks for the frequency of interest
    fft_coefs = np.zeros(n_chunks, dtype=complex)
    for ind, chunk in enumerate(chunks):
        fourier_coef = np.fft.fft(chunk * hann_window)
        fft_coefs[ind] = fourier_coef[fft_freqs_idx]

    # Compute lagged coherence across data segments
    lcs_num = 0
    for ind in range(n_chunks - 1):
        lcs_num += fft_coefs[ind] * np.conj(fft_coefs[ind + 1])
    lcs_denom = np.sqrt(np.sum(np.abs(fft_coefs[:-1])**2) * np.sum(np.abs(fft_coefs[1:])**2))

    # Normalize the lagged coherence value
    lc_val = np.abs(lcs_num / lcs_denom)

    return lc_val
