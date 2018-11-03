.. _api_documentation:

=================
API Documentation
=================

.. currentmodule:: neurodsp


Cycle-by-cycle features
=======================

.. autosummary::
   :toctree: generated/

   features.compute_features

Cycle-by-cycle segmentation
===========================

.. autosummary::
   :toctree: generated/

   cyclepoints.find_extrema
   cyclepoints.find_zerox

Waveform phase estimation
=========================

.. autosummary::
    :toctree: generated/

    cyclepoints.extrema_interpolated_phase

Burst detection
===============

.. autosummary::
    :toctree: generated/

    burst.detect_bursts_cycles
    burst.detect_bursts_df_amp
    burst.twothresh_amp
    burst.plot_burst_detect_params
