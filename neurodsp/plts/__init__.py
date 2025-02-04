"""Plotting functions."""

from .time_series import (plot_time_series, plot_bursts, plot_instantaneous_measure,
                          plot_multi_time_series)
from .filt import plot_filter_properties, plot_frequency_response, plot_impulse_response
from .rhythm import plot_swm_pattern, plot_lagged_coherence
from .spectral import (plot_power_spectra, plot_spectral_hist, plot_spectra_3d,
                       plot_scv, plot_scv_rs_lines, plot_scv_rs_matrix)
from .timefrequency import plot_timefrequency
from .aperiodic import plot_autocorr
from .combined import plot_timeseries_and_spectra
