"""
IIR Filters
===========

Design, apply, and evaluate IIR Filters.

This tutorial covers the design and application of Infinite Impulse Response (IIR) filters
(``neurodsp.filt.iir``).

"""
###################################################################################################

import numpy as np
from neurodsp.sim import sim_combined
from neurodsp.filt import filter_signal
from neurodsp.filt.iir import design_iir_filter, apply_iir_filter
from neurodsp.filt.utils import compute_frequency_response
from neurodsp.plts import plot_filter_properties, plot_time_series

###################################################################################################
# Introduction
# ------------
#
# In contrast to FIR filters, IIR filters produce an impulse response that is dependent on past and
# values of the impulse and impulse response (i.e. recursion or feedback), producing a response that
# is never zero. Due to this, the IIR filters are not typically implemented using convolution. The
# math introduced in the FIR tutorial may be extended to IIR filters, with a second expression
# representing feedback:
#
# .. math::
#
#    y(n) = \sum_{k=0}^M b_k x(n-k) - \sum_{k=1}^N a_k y(n-k)
#
# .. math::
#
#    H(z) = \frac{\sum_{k=0}^M b_k z^{-k}}{\sum_{k=1}^N a_k z^{-k}}
#


###################################################################################################
# Design
# ------
#
# Neurodsp supports the butterworth IIR filter. The order of this filter controls how smooth (low
# orders) or steep (high order) the roll-off of the transition band is. Below, we will design an
# IIR filter and plot the frquency response.
#

# Settings
fs = 1000
pass_type = 'bandpass'
f_range = (1, 50)
butterworth_order = 2

# Get the second-order series (sos) representatino of the filter
sos = design_iir_filter(fs, pass_type, f_range, butterworth_order)

###################################################################################################
# Apply
# -----
#
# The filter we previously designed may now be applied to a signal. Inspecting the filtered signal,
# relative to the original signal, is recommended.
#

# Simulate
n_seconds = 1

components = {'sim_powerlaw' : {'exponent' : 0},
              'sim_oscillation' : {'freq' : 10}}

variances = [0.1, 1]

sig = sim_combined(n_seconds, fs, components, variances)

# Apply the filter
sig_filt = apply_iir_filter(sig, sos)

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
# :func:`~.filter_signal` function. The ``plot_properties`` argument plots the frequency response.
# The frqeuency response and parameters may be saved by passing a path and filename to the
# ``save_properties`` argument.
#

sig_filt = filter_signal(sig, fs, pass_type, f_range, filter_type='iir',
                         butterworth_order=2, plot_properties=True, print_transitions=True)
