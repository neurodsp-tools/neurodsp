"""Spectral module, for calculating power spectra and associated functionality."""

from .spectral import (compute_spectrum, compute_spectrum_welch,
                       compute_spectrum_wavelet, compute_spectrum_medfilt)
from .spectral import compute_scv, compute_scv_rs, compute_spectral_hist
from .utils import trim_spectrum, rotate_powerlaw
