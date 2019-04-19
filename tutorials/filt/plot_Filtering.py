"""
Filtering
=========

Using digital filters on neural signals, including highpass, lowpass, bandpass & bandstop.

This tutorial primarily covers :mod:`neurodsp.filt`.
"""

###################################################################################################
#
# Sections
# --------
#
# This tutorial is broken down into the following sections:
#
# 1. Bandpass filter: extract a single oscillator from your signal
# 2. Highpass, lowpass, and bandstop filters: remove power in unwanted frequency ranges
# 3. Time-frequency resolution tradeoff: Change the filter length
# 4. Infinite-impulse-response (IIR) filter option.
# 5. Beta bandpass filter on a neural signal
#

###################################################################################################

import numpy as np

from neurodsp import filt
from neurodsp.utils import create_times
from neurodsp.plts.time_series import plot_time_series

###################################################################################################

# Set the random seed, for consistency simulating data
np.random.seed(0)

###################################################################################################
#
# 1. Bandpass filter
# ------------------
#
# Extract signal within a specific frequency range (e.g. theta, 4-8 Hz).
#

###################################################################################################

# Generate an oscillation with noise
fs = 1000
times = create_times(4, 1000)
sig = np.random.randn(len(times)) + 5*np.sin(times*2*np.pi*6)

###################################################################################################

# Filter the data, across a frequency band of interest
f_range = (4, 8)
sig_filt = filt.filter_signal(sig, fs, 'bandpass', f_range)

###################################################################################################

# Plot filtered signal
plot_time_series(times, [sig, sig_filt], ['Raw', 'Filtered'])

###################################################################################################
#
# Notice that the edges of the filtered signal are clipped (no red).
#
# Edge artifact removal is done by default in :func:`filter_signal`, because
# the signal samples at the edges only experienced part of the filter.
#
# To bypass this feature, set `remove_edge_artifacts=False`, but at your own risk!
#

###################################################################################################
#
# 2. Highpass, lowpass, and bandstop filters
# ------------------------------------------
#
# 2a. Highpass filter
# ~~~~~~~~~~~~~~~~~~~
#
# Remove low frequency drift from the data
#

###################################################################################################

# Generate a signal with a low-frequency drift
times = create_times(6, fs)
sig = np.random.randn(len(times)) + 5 * np.sin(times*2*np.pi*3) + 4 * np.sin(times*2*np.pi*.5)

###################################################################################################

# Filter the data
f_range = (2, None)
sig_filt = filt.filter_signal(sig, fs, 'highpass', f_range)

###################################################################################################

# Plot filtered signal
plot_time_series(times, [sig, sig_filt], ['Raw', 'Filtered'])

###################################################################################################
#
# 2b. Lowpass filter
# ~~~~~~~~~~~~~~~~~~
#
# Remove high frequency activity from the data.
#

###################################################################################################

# Generate a signal with a low-frequency drift
times = create_times(6, fs)
sig = np.random.randn(len(times)) + 5 * np.sin(times*2*np.pi*3) + 4 * np.sin(times*2*np.pi*.5)

###################################################################################################

# Filter the data
f_range = (None, 20)
sig_filt = filt.filter_signal(sig, fs, 'lowpass', f_range)

###################################################################################################

# Plot filtered signal
plot_time_series(times, [sig, sig_filt], ['Raw', 'Filtered'])

###################################################################################################
#
# 2c. Bandstop filter
# ~~~~~~~~~~~~~~~~~~~
#
# Remove 60Hz noise from the data
#
# Notice that it is necessary to set a nondefault filter length because
# a filter of length 3 cycles of a 58Hz oscillation would not attenuate
# the 60Hz oscillation much (try this yourself!).
#

###################################################################################################

# Generate a signal with a low-frequency drift
times = create_times(8, fs)
sig = 5 * np.sin(times*2*np.pi*5) + 2 * np.sin(times*2*np.pi*60)

###################################################################################################

# Filter the data
f_range = (58, 62)
sig_filt = filt.filter_signal(sig, fs, 'bandstop', f_range, n_seconds=0.5)

###################################################################################################

# Plot filtered signal
plot_time_series(times, [sig, sig_filt], ['Raw', 'Filtered'])

###################################################################################################

# Note the user warning above.
# This is because in the computed frequency response (below),
# the attenuation in the stopband does not go below 20dB.
sig_filt = filt.filter_signal(sig, fs, 'bandstop', f_range, n_seconds=0.25,
                              plot_properties=True)

###################################################################################################v

# This user warning disappears if we elongate the filter
sig_filt = filt.filter_signal(sig, fs, 'bandstop', f_range, n_seconds=1,
                              plot_properties=True)

###################################################################################################
#
# 3. Time-frequency resolution tradeoff
# -------------------------------------
#
# With longer filter kernels, we get improved frequency resolution,
# but worse time resolution.
#
# Two bandpass filters (one long and one short)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Notice that the short filter preserves the start of the oscillation
# better than the long filter (i.e. better time resolution).
#
# Notice that the long filter correctly removed the 1Hz oscillation,
# but the short filter did not (i.e. better frequency resolution).
#

###################################################################################################

# Generate an oscillation with noise
fs = 100
times = create_times(3, fs)
sig = np.random.randn(len(times)) * .3 + 5 * np.sin(times*2*np.pi*6) + 4 * np.sin(times*2*np.pi*1)

###################################################################################################

# Set the first second to 0
sig[:fs] = 0

# Define the frequency band of interest
f_range = (4, 8)

###################################################################################################

# Filter the data
sig_filt_short = filt.filter_signal(sig, fs, 'bandpass', f_range, n_seconds=.1)
sig_filt_long = filt.filter_signal(sig, fs, 'bandpass', f_range, n_seconds=1)

###################################################################################################

# Plot filtered signal
plot_time_series(times, [sig, sig_filt_short, sig_filt_long],
                 ['Raw', 'Short Filter', 'Long Filter'])

###################################################################################################

# Visualize the kernels and frequency responses
print('Short filter')
sig_filt_short = filt.filter_signal(sig, fs, 'bandpass', f_range, n_seconds=.1,
                                    plot_properties=True)
print('\n\nLong filter')
sig_filt_long = filt.filter_signal(sig, fs, 'bandpass', f_range, n_seconds=1,
                                   plot_properties=True)

###################################################################################################
#
# 4. Infinite impulse response (IIR) filter option
# ------------------------------------------------
#
# So far, the filters that we've been using are finite impulse response
# (FIR) filters. These filters are nice because we have good control over
# their properties (by manipulating the time-frequency resolution tradeoff
# through the filter length).
#
# However, sometimes we may not be as concerned with the precise filter
# properties, and so there is a faster option: IIR filters. We often use
# these filters when removing 60Hz line noise.
#
# Here we apply a 3rd order butterworth filter to remove 60Hz noise.
# Notice that some edge artifacts remain.
#

###################################################################################################

# Generate a signal with a low-frequency drift
fs = 1000
times = create_times(2, fs)
sig = 5 * np.sin(times*2*np.pi*5) + 2 * np.sin(times*2*np.pi*60)

###################################################################################################

# Low-pass filter the signal at 100Hz, just for fun.
sig = filt.filter_signal(sig, fs, 'lowpass', f_range=100)

###################################################################################################

# Filter the data
f_range = (58, 62)
sig_filt = filt.filter_signal(sig, fs, 'bandstop', f_range, filt_type='iir', butterworth_order=3)

###################################################################################################

# Plot filtered signal
plot_time_series(times, [sig, sig_filt], ['Raw', 'Filtered'], xlim=[0, 0.2])

###################################################################################################
#
# 5. Beta bandpass filter on neural signal
# ----------------------------------------

###################################################################################################

# Generate a signal with a low-frequency drift
sig = np.load('../data/sample_data_1.npy')
fs = 1000
times = create_times(len(sig)/fs, fs)

###################################################################################################

# Filter the data
# If you want to get rid of the transition band printouts, set verbose=False
f_range = (13, 30)
sig_filt, kernel = filt.filter_signal(sig, fs, 'bandpass', f_range, n_cycles=3,
                                      plot_properties=True, return_filter=True)

###################################################################################################

# Plot filtered signal
plot_time_series(times, [sig, sig_filt], ['Raw', 'Filtered'], xlim=[2, 5])

###################################################################################################
#
# Notice that in the filtered time series, the resulting oscillation appears to be
# more sinusoidal than the original signal really is.
#
# If you are interested in this problem, and how to deal with it, you should check out
# `bycycle <https://bycycle-tools.github.io/bycycle/>`_,
# which is a tool for time-domain analyses of waveform shape.
#

###################################################################################################
#
# Sphinx settings:
# sphinx_gallery_thumbnail_number = 12
#
