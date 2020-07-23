"""IRASA method."""

import fractions

import numpy as np

from scipy import signal
from scipy.optimize import curve_fit

from neurodsp.spectral import compute_spectrum, trim_spectrum

###################################################################################################
###################################################################################################

def irasa(sig, fs=None, f_range=(1, 30), hset=None, **spectrum_kwargs):
    """Separate the aperiodic and periodic components using the IRASA method.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        The sampling frequency of sig.
    f_range : tuple or None
        Frequency range.
    hset : 1d array
        Resampling factors used in IRASA calculation.
        If not provided, defautls to values from 1.1 to 1.9 with an increment of 0.05.
    spectrum_kwargs : dict
        Optional keywords arguments that are passed to `compute_spectrum`.

    Returns
    -------
    freqs : 1d array
        Frequency vector.
    psd_aperiodic : 1d array
        The aperiodic component of the power spectrum.
    psd_periodic : 1d array
        The periodic component of the power spectrum.

    Notes
    -----
    Irregular-Resampling Auto-Spectral Analysis (IRASA) is described in Wen & Liu (2016).
    Briefly, it aims to separate 1/f and periodic components by resampling time series, and
    computing power spectra, effectively averaging away any activity that is frequency specific.

    References
    ----------
    Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory Components in the Power Spectrum
    of Neurophysiological Signal. Brain Topography, 29(1), 13–26. DOI: 10.1007/s10548-015-0448-0
    """

    # Check & get hset. The rounding avoids floating precision errors
    hset = np.arange(1.1, 1.95, 0.05) if not hset else hset
    hset = np.round(hset, 4)

    # `nperseg` needs to be set to lock in the size of the FFT's
    if 'nperseg' not in spectrum_kwargs:
        spectrum_kwargs['nperseg'] = int(4 * fs)

    # Calculate the original spectrum across the whole signal
    freqs, psd = compute_spectrum(sig, fs, **spectrum_kwargs)

    # Do the IRASA resampling procedure
    psds = np.zeros((len(hset), *psd.shape))

    for ind, h in enumerate(hset):

        # Get the upsampling/downsampling (h, 1/h) factors as integer
        rat = fractions.Fraction(str(h))
        up, dn = rat.numerator, rat.denominator

        # Resample signal
        sig_up = signal.resample_poly(sig, up, dn, axis=-1)
        sig_dn = signal.resample_poly(sig, dn, up, axis=-1)

        # Calculate the PSD using same params as original
        freqs_up, psd_up = compute_spectrum(sig_up, h * fs, **spectrum_kwargs)
        freqs_dn, psd_dn = compute_spectrum(sig_dn, fs / h, **spectrum_kwargs)

        # Geometric mean of h and 1/h
        psds[ind, :] = np.sqrt(psd_up * psd_dn)

    # Now we take the median resampled spectra, as an estimate of the aperiodic component
    psd_aperiodic = np.median(psds, axis=0)

    # Subtract aperiodic from original, to get the periodic component
    psd_periodic = psd - psd_aperiodic

    # Restrict spectrum to requested range
    if f_range:
        psds = np.array([psd_aperiodic, psd_periodic])
        freqs, (psd_aperiodic, psd_periodic) = trim_spectrum(freqs, psds, f_range)

    return freqs, psd_aperiodic, psd_periodic


def fit_irasa(freqs, psd_aperiodic):
    """Fit the IRASA derived aperiodic component of the spectrum.

    Parameters
    ----------
    freqs : 1d array
        Frequency vector, in linear space.
    psd_aperidic : 1d array
        Power values, in linear space.

    Returns
    -------
    intercept : float
        Fit intercept value.
    slope : float
        Fit slope value.
    """

    popt, _ = curve_fit(fit_func, np.log(freqs), np.log(psd_aperiodic))
    intercept, slope = popt

    return intercept, slope


def fit_func(freqs, intercept, slope):
    return slope * freqs + intercept
