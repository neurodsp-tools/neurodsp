"""Utility function for neurodsp.spectral."""

import numpy as np
from scipy.fft import next_fast_len

# Alias a function that has moved, for backwards compatibility
from neurodsp.sim.modulate import rotate_spectrum as rotate_powerlaw

###################################################################################################
###################################################################################################

def trim_spectrum(freqs, power_spectra, f_range):
    """Extract a frequency range of interest from power spectra.

    Parameters
    ----------
    freqs : 1d array
        Frequency values for the power spectrum.
    power_spectra : 1d or 2d array
        Power spectral density values. If 2d, should be as [n_spectra, n_freqs].
    f_range : list of [float, float]
        Frequency range to restrict to, as [f_low, f_high].

    Returns
    -------
    freqs_ext : 1d array
        Extracted frequency values for the power spectrum.
    power_spectra_ext : 1d or 2d array
        Extracted power spectral density values.

    Notes
    -----
    This function extracts frequency ranges >= f_low and <= f_high.
    It does not round to below or above f_low and f_high, respectively.

    Examples
    --------
    Trim the power spectrum of a simulated time series:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.spectral import compute_spectrum
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs, spectrum = compute_spectrum(sig, fs=500)
    >>> freqs_ext, spec_ext = trim_spectrum(freqs, spectrum, [1, 30])
    """

    # Create mask to index only requested frequencies
    f_mask = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])

    # Restrict freqs & spectrum to requested range
    #   The if/else is to cover both 1d or 2d arrays
    freqs_ext = freqs[f_mask]
    power_spectra_ext = power_spectra[f_mask] if power_spectra.ndim == 1 \
        else power_spectra[:, f_mask]

    return freqs_ext, power_spectra_ext


def trim_spectrogram(freqs, times, spg, f_range=None, t_range=None):
    """Extract a frequency or time range of interest from a spectrogram.

    Parameters
    ----------
    freqs : 1d array
        Frequency values for the spectrogram.
    times : 1d array
        Time values for the spectrogram.
    spg : 2d array
        Spectrogram, or time frequency representation of a signal.
        Formatted as [n_freqs, n_time_windows].
    f_range : list of [float, float]
        Frequency range to restrict to, as [f_low, f_high].
    t_range : list of [float, float]
        Time range to restrict to, as [t_low, t_high].

    Returns
    -------
    freqs_ext : 1d array
        Extracted frequency values for the power spectrum.
    times_ext : 1d array
        Extracted segment time values
    spg_ext : 2d array
        Extracted spectrogram values.

    Notes
    -----
    This function extracts frequency ranges >= f_low and <= f_high,
    and time ranges >= t_low and <= t_high. It does not round to below
    or above f_low and f_high, or t_low and t_high, respectively.

    Examples
    --------
    Trim the spectrogram of a simulated time series:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.timefrequency import compute_wavelet_transform
    >>> from neurodsp.utils.data import create_times, create_freqs
    >>> fs = 500
    >>> n_seconds = 10
    >>> times = create_times(n_seconds, fs)
    >>> sig = sim_combined(n_seconds, fs,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs = create_freqs(1, 15)
    >>> mwt = compute_wavelet_transform(sig, fs, freqs)
    >>> spg = abs(mwt)**2
    >>> freqs_ext, times_ext, spg_ext = trim_spectrogram(freqs, times, spg,
    ...                                                  f_range=[8, 12], t_range=[0, 5])
    """

    # Initialize spg_ext, to define for case in which neither f_range nor t_range is defined
    spg_ext = spg

    # Restrict frequency range of the spectrogram
    if f_range is not None:
        f_mask = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
        freqs_ext = freqs[f_mask]
        spg_ext = spg_ext[f_mask, :]
    else:
        freqs_ext = freqs

    # Restrict time range of the spectrogram
    if t_range is not None:
        times_mask = np.logical_and(times >= t_range[0], times <= t_range[1])
        times_ext = times[times_mask]
        spg_ext = spg_ext[:, times_mask]
    else:
        times_ext = times

    return freqs_ext, times_ext, spg_ext


def get_positive_fft_outputs(freqs, powers=None, drop_zero=False):
    """Get the positive frequency values for an FFT.

    Parameters
    ----------
    freqs : 1d array
        Frequency vector corresponding to the FFT estimate, with positive & negative frequencies.
    powers : 1d array, optional
        Complex power value estimates from the FFT.
    drop_zero : bool, optional, default: False
        Whether to drop the estimate for frequency of 0.

    Returns
    -------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    spectrum : array
        Power spectral density.
        Only returned if an input power spectrum is passed.

    Notes
    -----
    This can be used to extract positive only frequency from an FFT, for example,
    as returned by `np.fft.fft` & np.fft.fftfreq`.
    """

    start_ind = 1 if drop_zero else 0

    # Get the max positive ind as half length, rounded up if the length is odd
    end_ind = int(np.ceil(len(freqs) / 2))

    if powers is not None:
        return freqs[start_ind:end_ind], powers[start_ind:end_ind]
    else:
        return freqs[start_ind:end_ind]


def pad_signal(sig, length):
    """Pad a signal to a desired length.

    Parameters
    ----------
    sig : 1d array
        Signal to pad.
    length : int
        Output length to pad the signal to.

    Returns
    -------
    sig : 1d array
        Padded signal.

    Notes
    -----
    This approach pads the signal evenly on the left and right side with 0s.
    If the padding length ends up being odd, this approach will split to padding to
    have one less at the front / left side pad, and one more on the right / end side pad.
    """

    if length > len(sig):
        npad_total = length - len(sig)
        npad_left, npad_right = int(np.floor(npad_total / 2)), int(np.ceil(npad_total / 2))
        sig = np.pad(sig, (npad_left, npad_right), mode='constant', constant_values=0)

    return sig


def window_pad(sig, nperseg, noverlap, npad, fast_len,
               nwindows=None, nsamples=None, pad_left=None, pad_right=None):
    """Pads windows (for Welch's PSD) with zeros.

    Parameters
    ----------
    sig : 1d or 2d array
        Time series.
    nperseg : int
        Length of each segment, in number of samples, at the beginning and end of each window.
    noverlap : int
        Number of points to overlap between segments, applied prior to zero padding.
    npad : int
        Number of samples to zero pad windows per side.
    fast_len : bool
        Moves nperseg to the fastest length to reduce computation.
        Adjusts zero-padding to account for the new nperseg.
        See scipy.fft.next_fast_len for details.
    nwindows, nsamples, pad_left, pad_right : int, optional, default: None
        Prevents redundant computation when sig is 2d.

    Returns
    -------
    sig_windowed : 1d or 2d array
        Windowed signal, with zeros padded at the around each window.
    """

    if sig.ndim == 2:

        # Determine nsamples & padding once, to prevent redundant computation in the loop
        nwindows = int(np.ceil(len(sig[0])/nperseg))
        if nsamples is None or pad_left is None or pad_right is None:
            nsamples, pad_left, pad_right = _find_pad_size(nperseg, npad, fast_len)

        # Recursively call window_pad on each signal
        for sind, csig in enumerate(sig):

            _sig_win, _nperseg, _noverlap = window_pad(
                # Required arguments
                csig, nperseg, noverlap, npad, fast_len,
                # Optional arguments to prevent redundant computation
                nwindows, nsamples, pad_left, pad_right,
            )

            if sind == 0:
                # Initialize windowed array
                sig_windowed = np.zeros((len(sig), len(_sig_win)))

            sig_windowed[sind] = _sig_win

        # Update nperseg and noverlap
        nperseg, noverlap = _nperseg, _noverlap

    else:

        # Compute the number of windows, samples, and padding.
        #   Do not recompute if called from the 2d case
        if nwindows is None:
            nwindows = int(np.ceil(len(sig) / nperseg))

        if nsamples is None or pad_left is None or pad_right is None:
            # Skipped if called from the 2d case
            nsamples, pad_left, pad_right = _find_pad_size(nperseg, npad, fast_len)

        # Window signal
        sig_windowed = np.zeros((nwindows, nsamples))

        for wind in range(nwindows):

            # Signal indices
            start = max(0, (wind * nperseg) - noverlap)
            end = min(len(sig), start + nperseg)

            if end - start != nperseg:
                # Stop if a full window can't be created at end of signal
                break

            # Pad
            sig_windowed[wind] = np.pad(sig[start:end], (pad_left, pad_right))

        # Removed incomplete windows and flatten
        sig_windowed = sig_windowed[:wind].flatten()

        # Update nperseg
        nperseg += (pad_left + pad_right)

        # Overlap is zero since overlapping segments was applied prior to padding each window
        noverlap = 0

    return sig_windowed, nperseg, noverlap


def _find_pad_size(nperseg, npad, fast_len):
    """Determine pad size and number of samples required."""

    nsamples = nperseg + npad

    pad_left = npad // 2
    pad_right = npad - pad_left

    if fast_len:
        # Increase nsamples to the next fastest length and update for zero-padding size
        nsamples = next_fast_len(nsamples)

        # New padding
        npad = nsamples - nperseg
        pad_left = npad // 2
        pad_right = npad - pad_left

    return nsamples, pad_left, pad_right
