.. _api_documentation:

=================
API Documentation
=================

This is the API reference for the neurodsp module.

Table of Contents
=================

.. contents::
   :local:
   :depth: 2

Filtering
---------

Functions and utilities for the :mod:`filt` module, for filtering time series.

General Filter Function
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.filt.filter

.. autosummary::
  :toctree: generated/

  filter_signal


FIR Filters
~~~~~~~~~~~

.. currentmodule:: neurodsp.filt.fir

.. autosummary::
  :toctree: generated/

  filter_signal_fir
  design_fir_filter

IIR Filters
~~~~~~~~~~~

.. currentmodule:: neurodsp.filt.iir

.. autosummary::
  :toctree: generated/

  filter_signal_iir
  design_iir_filter

Check Filter Properties
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.filt.checks

.. autosummary::
  :toctree: generated/

  check_filter_definition
  check_filter_properties

Filter Utilities
~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.filt.utils

.. autosummary::
  :toctree: generated/

  compute_frequency_response
  compute_pass_band
  compute_transition_band
  compute_nyquist
  remove_filter_edges

Time-Frequency Analyses
-----------------------

Functions and utilities in the :mod:`timefreq` module, for time-frequency analyses.

Hilbert Methods
~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.timefrequency.hilbert

.. autosummary::
  :toctree: generated/

  robust_hilbert
  phase_by_time
  amp_by_time
  freq_by_time

Wavelet Methods
~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.timefrequency.wavelets

.. autosummary::
  :toctree: generated/

  morlet_transform
  morlet_convolve

Spectral Analyses
-----------------

Functions and utilities in the :mod:`spectral` module, for spectral analyses.

Spectral Power Measures
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.spectral.spectral

.. autosummary::
   :toctree: generated/

  compute_spectrum
  compute_spectrum_welch
  compute_spectrum_wavelet
  compute_spectrum_medfilt

Spectral Variance Measures
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.spectral.spectral

.. autosummary::
   :toctree: generated/

  compute_scv
  compute_scv_rs
  compute_spectral_hist

Spectral Utilities
~~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.spectral.utils

.. autosummary::
   :toctree: generated/

  trim_spectrum
  rotate_powerlaw

Burst Detection
---------------

Functions and utilities in the :mod:`burst` module, for detection bursts in time series.

Burst Detection Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.burst

.. autosummary::
  :toctree: generated/

  detect_bursts_dual_threshold

Burst Utilities
~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.burst.utils

.. autosummary::
  :toctree: generated/

  compute_burst_stats

Rhythm Analyses
---------------

Functions and utilities in the :mod:`rhythm` module, for finding and analyzing rhythmic and recurring patterns in time series.

Sliding Window Matching
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.rhythm.swm

.. autosummary::
  :toctree: generated/

  sliding_window_matching

Lagged Coherence
~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.rhythm.lc

.. autosummary::
  :toctree: generated/

  lagged_coherence

Simulations
-----------

Functions and utilities in the :mod:`sim` module, for simulating time series.

Periodic Signals
~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.sim.periodic

.. autosummary::
  :toctree: generated/

  sim_oscillation
  sim_bursty_oscillation

Aperiodic Signals
~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.sim.aperiodic

.. autosummary::
  :toctree: generated/

  sim_powerlaw
  sim_poisson_pop
  sim_synaptic_current
  sim_random_walk

Transients
~~~~~~~~~~

.. currentmodule:: neurodsp.sim.transients

.. autosummary::
  :toctree: generated/

  sim_osc_cycle
  sim_synaptic_kernel

Combined Signals
~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.sim.combined

.. autosummary::
  :toctree: generated/

  sim_combined

Plots
-----

Functions for in the :mod:`plts` module, for plotting time series and analysis outputs.

Time Series
~~~~~~~~~~~

.. currentmodule:: neurodsp.plts.time_series

.. autosummary::
  :toctree: generated/

  plot_time_series
  plot_instantaneous_measure
  plot_bursts

Spectral
~~~~~~~~

.. currentmodule:: neurodsp.plts.spectral

.. autosummary::
  :toctree: generated/

  plot_power_spectra
  plot_scv
  plot_scv_rs_lines
  plot_scv_rs_matrix
  plot_spectral_hist

Filter
~~~~~~

.. currentmodule:: neurodsp.plts.filt

.. autosummary::
  :toctree: generated/

  plot_filter_properties
  plot_frequency_response
  plot_impulse_response

Rhythm
~~~~~~

.. currentmodule:: neurodsp.plts.rhythm

.. autosummary::
  :toctree: generated/

  plot_swm_pattern
  plot_lagged_coherence
