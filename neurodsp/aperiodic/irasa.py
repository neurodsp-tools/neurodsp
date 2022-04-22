"""The IRASA algorithm for separating periodic and aperiodic activity."""

import fractions

import numpy as np

from scipy import signal
from scipy.optimize import curve_fit

from neurodsp.spectral import compute_spectrum, trim_spectrum

###################################################################################################
###################################################################################################

def compute_irasa(sig, fs, f_range=None, hset=None, thresh=None, **spectrum_kwargs):
    """Separate aperiodic and periodic components using IRASA.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        The sampling frequency of sig.
    f_range : tuple, optional
        Frequency range to restrict the analysis to.
    hset : 1d array, optional
        Resampling factors used in IRASA calculation.
        If not provided, defaults to values from 1.1 to 1.9 with an increment of 0.05.
    thresh : float, optional
        A relative threshold to apply when separating out periodic components.
        The threshold is defined in terms of standard deviations of the original spectrum.
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
    Irregular-Resampling Auto-Spectral Analysis (IRASA) is an algorithm ([1]_) that aims to
    separate 1/f and periodic components by resampling time series and computing power spectra,
    averaging away any activity that is frequency specific to isolate the aperiodic component.

    References
    ----------
    .. [1] Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory Components in
           the Power Spectrum of Neurophysiological Signal. Brain Topography, 29(1), 13–26.
           DOI: https://doi.org/10.1007/s10548-015-0448-0

    Examples
    --------
    Apply IRASA to a simulated combined time series:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation': {'freq': 10}})
    >>> freqs, psd_aperiodic, psd_periodic = compute_irasa(sig, fs=500, f_range=[3, 50])
    """

    # Check & get the resampling factors, with rounding to avoid floating point precision errors
    hset = np.arange(1.1, 1.95, 0.05) if hset is None else hset
    hset = np.round(hset, 4)

    # The `nperseg` input needs to be set to lock in the size of the FFT's
    if 'nperseg' not in spectrum_kwargs:
        spectrum_kwargs['nperseg'] = int(4 * fs)

    # Calculate the original spectrum across the whole signal
    freqs, psd = compute_spectrum(sig, fs, **spectrum_kwargs)

    # Do the IRASA resampling procedure
    psds = np.zeros((len(hset), *psd.shape))
    for ind, h_val in enumerate(hset):

        # Get the up-sampling / down-sampling (h, 1/h) factors as integers
        rat = fractions.Fraction(str(h_val))
        up, dn = rat.numerator, rat.denominator

        # Resample signal
        sig_up = signal.resample_poly(sig, up, dn, axis=-1)
        sig_dn = signal.resample_poly(sig, dn, up, axis=-1)

        # Calculate the power spectrum, using the same params as original
        freqs_up, psd_up = compute_spectrum(sig_up, h_val * fs, **spectrum_kwargs)
        freqs_dn, psd_dn = compute_spectrum(sig_dn, fs / h_val, **spectrum_kwargs)

        # Calculate the geometric mean of h and 1/h
        psds[ind, :] = np.sqrt(psd_up * psd_dn)

    # Take the median resampled spectra, as an estimate of the aperiodic component
    psd_aperiodic = np.median(psds, axis=0)

    # Subtract aperiodic from original, to get the periodic component
    psd_periodic = psd - psd_aperiodic

    # Apply a relative threshold for tuning which activity is labeled as periodic
    if thresh is not None:
        sub_thresh = np.where(psd_periodic - psd_aperiodic < thresh * np.std(psd))[0]
        psd_periodic[sub_thresh] = 0
        psd_aperiodic[sub_thresh] = psd[sub_thresh]

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

    Notes
    -----
    This fits a linear function of the form `y = ax + b` to the log-log aperiodic power spectrum.
    """

    popt, _ = curve_fit(fit_func, np.log10(freqs), np.log10(psd_aperiodic))
    intercept, slope = popt

    return intercept, slope


def fit_func(freqs, intercept, slope):
    """A fit function to use for fitting IRASA separated 1/f power spectra components."""

    return slope * freqs + intercept
