"""Spectral module, for calculating power spectra, spectral variance, etc."""

from .power import (compute_spectrum, compute_spectrum_welch,
                    compute_spectrum_wavelet, compute_spectrum_medfilt)
from .variance import compute_scv, compute_scv_rs, compute_spectral_hist
from .utils import trim_spectrum, trim_spectrogram, rotate_powerlaw
