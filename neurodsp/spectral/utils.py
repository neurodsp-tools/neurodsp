"""Utility function for neurodsp.spectral."""

import numpy as np

###################################################################################################
###################################################################################################

def trim_spectrum(freqs, power_spectra, f_range):
    """Extract frequency range of interest from power spectra.

    Parameters
    ----------
    freqs : 1d array
        Frequency values for the PSD.
    power_spectra : 1d or 2d array
        Power spectral density values.
    f_range: list of [float, float]
        Frequency range to restrict to.

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
    """

    # Create mask to index only requested frequencies
    f_mask = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])

    # Restrict freqs & psd to requested range. The if/else is to cover both 1d or 2d arrays
    freqs_ext = freqs[f_mask]
    power_spectra_ext = power_spectra[f_mask] if power_spectra.ndim == 1 \
        else power_spectra[:, f_mask]

    return freqs_ext, power_spectra_ext


def rotate_powerlaw(freqs, spectrum, delta_exponent, f_rotation=1):
    """Change the power law exponent of a power spectrum about an axis frequency.

    Parameters
    ----------
    freqs : 1d array
        Frequency axis of input spectrum, in Hz. Must be same length as spectrum.
    spectrum : 1d array
        Power spectrum to be rotated.
    delta_exponent : float
        Change in power law exponent to be applied.
        Positive is counterclockwise rotation (flatten), negative is clockwise rotation (steepen).
    f_rotation : float, optional, default=1
        Axis of rotation frequency, in Hz, such that power at that frequency is unchanged
        by the rotation. Only matters if not further normalizing signal variance.

    Returns
    -------
    rot_spectrum : 1d array
        Rotated spectrum.
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
