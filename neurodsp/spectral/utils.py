"""Utility function for neurodsp.spectral."""

import numpy as np

###################################################################################################
###################################################################################################

def trim_spectrum(freqs, power_spectra, f_range):
    """Extract a frequency range of interest from power spectra.

    Parameters
    ----------
    freqs : 1d array
        Frequency values for the power spectrum.
    power_spectra : 1d or 2d array
        Power spectral density values.
    f_range: list of [float, float]
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

    # Restrict freqs & psd to requested range. The if/else is to cover both 1d or 2d arrays
    freqs_ext = freqs[f_mask]
    power_spectra_ext = power_spectra[f_mask] if power_spectra.ndim == 1 \
        else power_spectra[:, f_mask]

    return freqs_ext, power_spectra_ext


def rotate_powerlaw(freqs, spectrum, delta_exponent, f_rotation=1):
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
        Frequency at which to rotate the spectrum, where power is unchanged by the rotation, in Hz.
        This only matters if not further normalizing signal variance.

    Returns
    -------
    rotated_spectrum : 1d array
        Rotated spectrum.

    Examples
    --------
    Rotate a power spectrum, calculated on simualated data:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.spectral import compute_spectrum
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs, spectrum = compute_spectrum(sig, fs=500)
    >>> rotated_spectrum = rotate_powerlaw(freqs, spectrum, -1)
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
