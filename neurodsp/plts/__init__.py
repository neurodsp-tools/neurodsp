"""Plotting functions."""

from .time_series import plot_time_series, plot_bursts, plot_instantaneous_measure
from .filt import plot_filter_properties, plot_frequency_response, plot_impulse_response
from .rhythm import plot_swm_pattern, plot_lagged_coherence
from .spectral import (plot_power_spectra, plot_spectral_hist,
                       plot_scv, plot_scv_rs_lines, plot_scv_rs_matrix)
