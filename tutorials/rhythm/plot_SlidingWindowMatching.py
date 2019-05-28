"""
Sliding Window Matching
=======================

Find recurrent patterns in a neural signal using Sliding Window Matching.

This tutorial primarily covers :mod:`neurodsp.rhythm.swm`.
"""

###################################################################################################
#
# Overview
# --------
#
# This notebook shows how to use sliding window matching (SWM) for identifying recurring
# patterns in a neural signal, like the shape of an oscillatory waveform.
#
# For more details on Sliding Window Matching see Gips et al., 2017, J Neuro Methods.
#

###################################################################################################

import numpy as np

from neurodsp.utils import create_times
from neurodsp.rhythm import sliding_window_matching
from neurodsp.plts.rhythm import plot_swm_pattern
from neurodsp.plts.time_series import plot_time_series

###################################################################################################

# Set the random seed, for consistency simulating data
np.random.seed(0)

###################################################################################################
#
# Load neural signal
# ------------------
#

###################################################################################################

# Load example data
sig = np.load('../data/sample_data_1.npy')
fs = 1000
times = create_times(len(sig)/fs, fs)
f_range = (13, 30)

# Plot example signal
plot_time_series(times, sig)

###################################################################################################
#
# Apply sliding window matching to neural signal
# ----------------------------------------------
#
# Because we define the window length to be about 1 cycle, this should roughly extract
# the waveform shape of the neural oscillation. Notice that the beta cycles have sharper
# troughs than peaks, and the average window is a beta cycle with a sharp trough.
#
# However, notice that these results change dramatically by changing the random seed.
# Using more data and increasing the number of iterations helps the robustness of the algorithm.
#

###################################################################################################

# Define window length & minimum window spacing, both in seconds
win_len = .055
win_spacing = .2

# Apply the sliding window matching algorithm to the time series
avg_window, window_starts, J = sliding_window_matching(sig, fs, win_len, win_spacing,
                                                       max_iterations=500)

###################################################################################################

# Plot the discovered pattern
plot_swm_pattern(avg_window)

###################################################################################################
#
# Sphinx settings:
# sphinx_gallery_thumbnail_number = 2
#
