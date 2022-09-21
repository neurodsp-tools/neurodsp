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

Functions and utilities in the ``filt`` module, for filtering time series.

General Filter Function
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.filt
.. autosummary::
    :toctree: generated/

    filter_signal

FIR Filters
~~~~~~~~~~~

.. currentmodule:: neurodsp.filt
.. autosummary::
    :toctree: generated/

    filter_signal_fir
    design_fir_filter

IIR Filters
~~~~~~~~~~~

.. currentmodule:: neurodsp.filt
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

Functions and utilities in the ``timefrequency`` module, for time-frequency analyses.

Hilbert Methods
~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.timefrequency
.. autosummary::
    :toctree: generated/

    robust_hilbert
    phase_by_time
    amp_by_time
    freq_by_time

Wavelet Methods
~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.timefrequency
.. autosummary::
  :toctree: generated/

  compute_wavelet_transform
  convolve_wavelet

Spectral Analyses
-----------------

Functions and utilities in the ``spectral`` module, for spectral analyses.

Spectral Power
~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.spectral
.. autosummary::
    :toctree: generated/

    compute_spectrum
    compute_spectrum_welch
    compute_spectrum_wavelet
    compute_spectrum_medfilt

Spectral Measures
~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.spectral
.. autosummary::
    :toctree: generated/

    compute_absolute_power
    compute_relative_power
    compute_band_ratio

Spectral Variance
~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.spectral
.. autosummary::
    :toctree: generated/

    compute_scv
    compute_scv_rs
    compute_spectral_hist

Spectral Utilities
~~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.spectral
.. autosummary::
    :toctree: generated/

    trim_spectrum
    trim_spectrogram

Burst Detection
---------------

Functions and utilities in the ``burst`` module, for detection bursts in time series.

Burst Detection Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.burst
.. autosummary::
    :toctree: generated/

    detect_bursts_dual_threshold

Burst Utilities
~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.burst
.. autosummary::
    :toctree: generated/

    compute_burst_stats

Rhythm Analyses
---------------

Functions and utilities in the ``rhythm`` module, for finding and analyzing rhythmic and recurring patterns in time series.

Sliding Window Matching
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.rhythm
.. autosummary::
    :toctree: generated/

    sliding_window_matching

Lagged Coherence
~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.rhythm
.. autosummary::
    :toctree: generated/

    compute_lagged_coherence

Aperiodic Analyses
------------------

Functions and utilities in the ``aperiodic`` module, for analyzing aperiodic activity in time series.

Auto-correlation Analyses
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.aperiodic.autocorr
.. autosummary::
    :toctree: generated/

    compute_autocorr

Fluctuation Analyses
~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.aperiodic.dfa
.. autosummary::
    :toctree: generated/

    compute_fluctuations
    compute_rescaled_range
    compute_detrended_fluctuation

Signal Decomposition
~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.aperiodic.irasa
.. autosummary::
    :toctree: generated/

    compute_irasa
    fit_irasa

Conversions
~~~~~~~~~~~

.. currentmodule:: neurodsp.aperiodic.conversions
.. autosummary::
    :toctree: generated/

    convert_exponent_alpha
    convert_alpha_exponent
    convert_exponent_hurst
    convert_hurst_exponent
    convert_exponent_hfd
    convert_hfd_exponent

Simulations
-----------

Functions and utilities in the ``sim`` module, for simulating time series.

Periodic Signals
~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.sim
.. autosummary::
    :toctree: generated/

    sim_oscillation
    sim_bursty_oscillation
    sim_variable_oscillation
    sim_damped_oscillation

Aperiodic Signals
~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.sim
.. autosummary::
    :toctree: generated/

    sim_powerlaw
    sim_poisson_pop
    sim_synaptic_current
    sim_knee
    sim_random_walk
    sim_frac_gaussian_noise
    sim_frac_brownian_motion

Transients
~~~~~~~~~~

.. currentmodule:: neurodsp.sim.transients
.. autosummary::
    :toctree: generated/

    sim_synaptic_kernel
    sim_action_potential
    sim_damped_erp

Cycles
~~~~~~

.. currentmodule:: neurodsp.sim.cycles
.. autosummary::
    :toctree: generated/

    sim_cycle
    sim_sine_cycle
    sim_asine_cycle
    sim_sawtooth_cycle
    sim_gaussian_cycle
    sim_skewed_gaussian_cycle
    sim_exp_cos_cycle
    sim_asym_harmonic_cycle

Combined Signals
~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.sim.combined
.. autosummary::
    :toctree: generated/

    sim_combined
    sim_peak_oscillation
    sim_modulated_signal

Utilities
~~~~~~~~~

.. currentmodule:: neurodsp.sim.utils
.. autosummary::
    :toctree: generated/

    rotate_spectrum
    rotate_timeseries
    modulate_signal

Random Seed
~~~~~~~~~~~

.. currentmodule:: neurodsp.sim
.. autosummary::
    :toctree: generated/

    set_random_seed

Plots
-----

Functions in the ``plts`` module, for plotting time series and analysis outputs.

Time Series
~~~~~~~~~~~

.. currentmodule:: neurodsp.plts

.. autosummary::
    :toctree: generated/

    plot_time_series
    plot_instantaneous_measure
    plot_bursts

Spectral
~~~~~~~~

.. currentmodule:: neurodsp.plts
.. autosummary::
    :toctree: generated/

    plot_power_spectra
    plot_scv
    plot_scv_rs_lines
    plot_scv_rs_matrix
    plot_spectral_hist

Filter
~~~~~~

.. currentmodule:: neurodsp.plts
.. autosummary::
    :toctree: generated/

    plot_filter_properties
    plot_frequency_response
    plot_impulse_response

Rhythm
~~~~~~

.. currentmodule:: neurodsp.plts
.. autosummary::
    :toctree: generated/

    plot_swm_pattern
    plot_lagged_coherence

Time Frequency
~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.plts
.. autosummary::
    :toctree: generated/

    plot_timefrequency

Utilities
---------

Functions in the ``utils`` module, providing general utilities.

Normalization
~~~~~~~~~~~~~

.. currentmodule:: neurodsp.utils.norm
.. autosummary::
    :toctree: generated/

    normalize_sig
    demean
    normalize_variance

Data
~~~~

.. currentmodule:: neurodsp.utils.data
.. autosummary::
    :toctree: generated/

    create_freqs
    create_times
    create_samples
    compute_nsamples
    compute_nseconds
    compute_cycle_nseconds
    split_signal

Outliers
~~~~~~~~

.. currentmodule:: neurodsp.utils.outliers
.. autosummary::
    :toctree: generated/

    remove_nans
    restore_nans
    discard_outliers
