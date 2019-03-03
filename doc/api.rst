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

.. currentmodule:: neurodsp.filt

Design & Apply Filters
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: generated/

  filter_signal
  filter_signal_fir
  filter_signal_iir
  design_fir_filter
  design_iir_filter

Filter Properties
~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: generated/

  check_filter_definition
  check_filter_properties
  compute_frequency_response
  compute_pass_band
  compute_trans_band
  compute_nyquist

Time-Frequency analysis
-----------------------

.. currentmodule:: neurodsp.timefrequency

.. autosummary::
  :toctree: generated/

  phase_by_time
  amp_by_time
  freq_by_time

Spectral analysis
-----------------

.. currentmodule:: neurodsp.spectral

.. autosummary::
   :toctree: generated/

   compute_spectrum
   compute_scv
   compute_scv_rs
   spectral_hist
   morlet_transform
   morlet_convolve
   rotate_powerlaw

Rhythmic analysis
-----------------

Burst Detection
~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.burst

.. autosummary::
  :toctree: generated/

  detect_bursts_dual_threshold
  compute_burst_stats

Sliding Window Matching
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.swm

.. autosummary::
  :toctree: generated/

  sliding_window_matching

Lagged Coherence
~~~~~~~~~~~~~~~~

.. currentmodule:: neurodsp.laggedcoherence

.. autosummary::
  :toctree: generated/

  lagged_coherence

Signal simulation
-----------------

.. currentmodule:: neurodsp.sim

.. autosummary::
  :toctree: generated/

Periodic Signals
~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: generated/

  sim_oscillator
  sim_bursty_oscillator
  sim_jittered_oscillator

Aperiodic Signals
~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: generated/

  sim_filtered_noise
  sim_synaptic_noise
  sim_ou_process
  sim_variable_powerlaw

Combined Signals
~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: generated/

  sim_noisy_oscillator
  sim_noisy_bursty_oscillator
