"""
Burst Detection
===============

Analyze neural signals for bursts of oscillations of interest.

This tutorial primarily covers :mod:`neurodsp.burst`.
"""

###################################################################################################

import numpy as np

from neurodsp.utils import create_times
from neurodsp.sim.combined import sim_combined
from neurodsp.burst import detect_bursts_dual_threshold, compute_burst_stats

from neurodsp.plts.time_series import plot_time_series, plot_bursts

###################################################################################################

# Set the random seed, for consistency simulating data
np.random.seed(0)

###################################################################################################
#
# Simulate a noisy and bursty oscillator
# --------------------------------------
#
# First, we'll simulate bursting activity in the alpha range on top of an aperiodic component.
#

###################################################################################################

n_seconds = 5
fs = 1000
components = {'sim_synaptic_current' : {'n_neurons':1000, 'firing_rate':2,
                                        't_ker':1.0, 'tau_r':0.002, 'tau_d':0.02},
              'sim_bursty_oscillation' : {'freq' : 10,
                                          'prob_enter_burst' : .2, 'prob_leave_burst' : .2}}

# Simulate a signal with a bursty oscillation with an aperiodic component & a time vector
sig = sim_combined(n_seconds, fs, components)
times = create_times(n_seconds, fs)

###################################################################################################

plot_time_series(times, sig, 'Simulated EEG')

###################################################################################################
#
# We can see that 10Hz oscillations present 1.5-2.1 seconds and after 4.2 seconds.
#

###################################################################################################
#
# Dual amplitude threshold algorithm
# -----------------------------------
#
# This algorithm first computes the amplitude at each point in time for
# a given frequency range. This amplitude is then normalized by the average
# (default: median) amplitude of the whole time series. Two thresholds are
# defined based off of this normalized amplitude. In order for a burst to be
# detected, the amplitude must cross the higher amplitude threshold. The burst
# lasts until the amplitude then falls below the lower amplitude threshold.
#
# **Other Parameters:**
# * The average for normalization can be set to either the mean or median
# by modifying the `average_method` keyword argument.
# * Power can be used instead of amplitude by modifying the `magnitude_type`
# keyword argument.
#

# Detect bursts using 'deviation' algorithm
amp_dual_thresh = (1, 2)
f_range = (8, 12)
bursting = detect_bursts_dual_threshold(sig, fs, f_range, amp_dual_thresh)

# Plot original signal and burst activity
plot_bursts(times, sig, bursting, labels=['Simulated EEG', 'Detected Burst'])

###################################################################################################
#
# The graph above shows the bursting activity in red. The algorithm was
# used with thresh=(1, 2), so any time point with more than 3 times the
# median magnitude in the alpha range (8-12 Hz) was marked as bursting activity.

###################################################################################################
#
# Burst detection applied to real recordings
# ------------------------------------------

# Load data
sig = np.load('../data/sample_data_1.npy')
fs = 1000

# Same code as earlier
times = create_times(len(sig)/fs, fs)

# Detect bursts using 'deviation' algorithm
bursting = detect_bursts_dual_threshold(sig, fs, f_range, dual_thresh=(3, 3))

# Plot original signal and burst activity
plot_bursts(times, sig, bursting, labels=['Simulated EEG', 'Detected Burst'])

###################################################################################################
#
# No bursts were detected! There could be a number of reasons for this.
# One of the easy things to do is to adjust the parameters for burst detection.
# Let's try making the thresholds lower.
#

###################################################################################################

# Detect bursts using 'deviation' algorithm
bursting = detect_bursts_dual_threshold(sig, fs, f_range, dual_thresh=(1, 2))

# Plot original signal and burst activity
plot_bursts(times, sig, bursting, labels=['Simulated EEG', 'Detected Burst'])

###################################################################################################
#
# Better, but not very good. Note the undetected cycles just before 9s.
#
# There's another serious issue: we're looking for alpha bursts,
# but the bursts in this data are *beta* bursts.
#

###################################################################################################

# Detect bursts
bursting = detect_bursts_dual_threshold(sig, fs, (13, 30), dual_thresh=(1, 2))

# Plot original signal and burst activity
plot_bursts(times, sig, bursting, labels=['Simulated EEG', 'Detected Burst'])

###################################################################################################
#
# Much better! This just goes to show that burst detection is an art that
# requires some knowledge of the data you're working with.
#

###################################################################################################
#
# Sphinx settings:
# sphinx_gallery_thumbnail_number = 2
#
