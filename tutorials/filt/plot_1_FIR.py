"""
FIR Filters
===========

Design, apply, and evaluate FIR Filters.

This tutorial covers the design and application of Finite Impulse Response (FIR) filters
(``neurodsp.filt.fir``).

"""
###################################################################################################

import numpy as np
from neurodsp.sim import sim_combined
from neurodsp.filt import filter_signal
from neurodsp.filt.fir import design_fir_filter, apply_fir_filter
from neurodsp.filt.utils import compute_frequency_response
from neurodsp.plts import plot_filter_properties, plot_time_series

###################################################################################################
# Introduction
# ------------
#
# An impulse response is the output of the convolution of a short duration signal (i.e. Kronecker
# delta function) with the filter. FIR filters convolve an input signal with the impulse response.
# Each timepoint in the output signal is expressed as:
#
# ..math::
#
#   y(n) = \sum_{k=0}^M b_k x(n-k) - \sum_{k=1}^N a_k y(n-k)
#
# Where each output (:math:`y(n)`) is the sum of the product of the filter coefficients
# (i.e. values from the impulse response, :math:`b_k`) and past values of the input signal
# (:math:`x(n-k)``). The filter order or the length of the impulse response minus one is
# represented by :math:`M`.
#
# This formula also has a frequency representation:
#
# ..math::
#
#   H(z) = \sum_{k=0}^Mb_kz^{-k}
#

###################################################################################################
# Design
# ------
#
# Although filtering can increase the signal-to-noise ratio (SNR), improper filter design may result
# in the introduction of confounding distortions and artifacts. To help guard against this,
# `Widman et al., 2015 <https://pubmed.ncbi.nlm.nih.gov/25128257/>`_ recommends:
#
# 1. Considering the frequency and impulse response
# 2. Using simulation data to understand filter effects
# 3. Manually inspecting the filtered signal.
#
# The impulse and frequency response (:math:`b_k`) are found below for an alpha bandpass filter.
#

# Settings
fs = 1000
pass_type = 'bandpass'
f_range = (8, 13)

# Compute the impulse response
filter_coefs = design_fir_filter(fs, pass_type, f_range, n_cycles=3)

# Compute the frequency response
f_db, db = compute_frequency_response(filter_coefs, 1, fs)

# Plot
plot_filter_properties(f_db, db, fs, filter_coefs)

###################################################################################################
# Apply
# -----
#
# The filter we previously designed is applied to a simulated signal using convolution. Inspecting
# the filtered signal, relative to the original signal, is recommended.
#

# Simulate
n_seconds = 1

components = {'sim_powerlaw' : {'exponent' : 0},
              'sim_oscillation' : {'freq' : 10}}

variances = [0.1, 1]

sig = sim_combined(n_seconds, fs, components, variances)

# Apply the filter
sig_filt = apply_fir_filter(sig, filter_coefs)

# Plot
times = np.arange(0, len(sig)/fs, 1/fs)
plot_time_series(times, [sig, sig_filt], ['Raw', 'Filtered'])

###################################################################################################
# Reporting
# ---------
#
# In addition, to careful filter design, `Widman et al., 2015 <https://pubmed.ncbi.nlm.nih.gov/25128257/>`_
# recommends reporting the following filter parameters:
#
# - Filter Pass-Type
# - Cutoff Frequency and Definition
# - Filter Order
# - Transition Bandwidth
# - Passband Ripple and Stopband Attenuation
# - Filter Delay and Causality
# - Direction of Computation
#
# The design, application, and evaluation of a filter may be performed using the
# :func:`~.filter_signal` function. The ``verbose`` argument prints the filter parameters to the
# console. Alternatively, these parameters may be saved by passing a path and filename to the
# ``save_properties`` argument. Futhermore, the ``plot_properties`` argument plots the impulse and
# frequency response.
#

sig_filt = filter_signal(sig, fs, pass_type, f_range, filter_type='fir',
                         plot_properties=True, verbose=True)
