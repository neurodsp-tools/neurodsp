"""
Time-frequency analysis
=======================

Estimate instantaneous measures of phase, amplitude, and frequency.

This tutorial primarily covers :mod:`neurodsp.timefrequency`.
"""

###################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.utils import create_times
from neurodsp.timefrequency import amp_by_time, freq_by_time, phase_by_time
from neurodsp.plts.time_series import plot_time_series, plot_instantaneous_measure

###################################################################################################
#
# Load example neural signal
# --------------------------
#
# First, we will load an example neural signal to use for our time-frequency measures.
#

###################################################################################################

# Load a neural signal, as well as a filtered version of the same signal
sig = np.load('../data/sample_data_1.npy')
sig_filt_true = np.load('../data/sample_data_1_filt.npy')

# Set the sampling rate and create a times vector for the signal
fs = 1000
times = create_times(len(sig)/fs, fs)

# Set the frequency range to be used
f_range = (13, 30)

###################################################################################################

# Plot signal
plot_time_series(times, sig)

###################################################################################################
#
# Instantaneous Phase
# -------------------
#
# Instantaneous phase is a measure of the phase of a signal, over time.
#

###################################################################################################

# Compute instaneous phase from a signal
pha = phase_by_time(sig, fs, f_range)

###################################################################################################

# Plot example signal
_, axs = plt.subplots(2, 1, figsize=(15, 6))
plot_time_series(times, sig, xlim=[4, 5], xlabel=None, ax=axs[0])
plot_instantaneous_measure(times, pha, xlim=[4, 5], ax=axs[1])

###################################################################################################
#
# Instantaneous Amplitude
# -----------------------
#
# Instantaneous amplitude is a measure of the amplitude of a signal, over time.
#

###################################################################################################

# Compute instaneous amplitude from a signal
amp = amp_by_time(sig, fs, f_range)

###################################################################################################

# Plot example signal
_, axs = plt.subplots(2, 1, figsize=(15, 6))
plot_instantaneous_measure(times, [sig, amp], 'amplitude',
                           labels=['Raw Voltage', 'Amplitude'],
                           xlim=[4, 5], xlabel=None, ax=axs[0])
plot_instantaneous_measure(times, [sig_filt_true, amp], 'amplitude',
                           labels=['Raw Voltage', 'Amplitude'], colors=['b', 'r'],
                           xlim=[4, 5], ax=axs[1])

###################################################################################################
#
# Instantaneous Frequency
# -----------------------
#
# Instantaneous frequency is a measure of frequency across time.
#
# It is measured as the temporal derivative of the instantaneous phase.
#
# Intstantaneous frequency measures can exhibit abrupt shifts. Sometimes, a transform,
# such as appling a median filter, is used to make it smoother.
#
# For example of this, see Samaha & Postle, 2015.
#

###################################################################################################

# Compute instaneous frequency from a signal
i_f = freq_by_time(sig, fs, f_range)

###################################################################################################

# Plot example signal
_, axs = plt.subplots(3, 1, figsize=(15, 9))
plot_time_series(times, sig, 'Raw Voltage', xlim=[4, 5], xlabel=None, ax=axs[0])
plot_time_series(times, sig_filt_true,
                 labels='Beta Filtered Voltage', colors='b',
                 xlim=[4, 5], xlabel=None, ax=axs[1])
plot_instantaneous_measure(times, i_f, 'frequency', colors='r',
                           xlim=[4, 5], ylim=[10, 30], ax=axs[2])

###################################################################################################
#
# Sphinx settings:
# sphinx_gallery_thumbnail_number = 3
#
