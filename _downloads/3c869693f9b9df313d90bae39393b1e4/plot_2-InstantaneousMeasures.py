"""
Time-frequency analysis
=======================

Estimate instantaneous measures of phase, amplitude, and frequency.

This tutorial primarily covers :mod:`neurodsp.timefrequency`.
"""

###################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.timefrequency import amp_by_time, freq_by_time, phase_by_time
from neurodsp.plts.time_series import plot_time_series, plot_instantaneous_measure

###################################################################################################
#
# Load example neural signal
# --------------------------

sig = np.load('./data/sample_data_1.npy')
fs = 1000
times = np.arange(0, len(sig)/fs, 1/fs)
f_range = (13, 30)

# Load filtered version of signal
sig_filt_true = np.load('./data/sample_data_1_filt.npy')

###################################################################################################

# Plot signal
plot_time_series(times, sig)

###################################################################################################
#
# Compute and plot instantaneous phase
# ------------------------------------

###################################################################################################

# Compute instaneous phase from a signal
pha = phase_by_time(sig, fs, f_range)

###################################################################################################

# Plot example signal
_, axs = plt.subplots(2, 1, figsize=(15, 6))
plot_time_series(times, sig, xlim=[4, 5], ax=axs[0])
plot_instantaneous_measure(times, pha, xlim=[4, 5], ax=axs[1])


# plt.figure(figsize=(12, 4))
# plt.subplot(2, 1, 1)
# plt.plot(times, sig, 'k')
# plt.xlim((4, 5))
# plt.ylabel('Voltage (uV)')
# plt.subplot(2, 1, 2)
# plt.plot(times, pha, 'k')
# plt.xlim((4, 5))
# plt.yticks([-np.pi, 0, np.pi], ['$\pi$', 0, '$\pi$'])
# plt.xlabel('Time (s)')
# plt.ylabel('Phase (rad)')

###################################################################################################
#
# Compute and plot instantaneous amplitude
# ----------------------------------------

###################################################################################################

# Compute instaneous amplitude from a signal
amp = amp_by_time(sig, fs, f_range)

###################################################################################################

# Plot example signal
plt.figure(figsize=(12, 4))
plt.subplot(2, 1, 1)
plt.plot(times, sig, 'k', label='raw voltage')
plt.plot(times, amp, 'r', label='amplitude')
plt.legend(loc='best')
plt.xlim((4, 5))
plt.ylabel('Voltage (uV)')
plt.subplot(2, 1, 2)
plt.plot(times, sig_filt_true, 'b', label='beta-filtered voltage')
plt.plot(times, amp, 'r', label='amplitude')
plt.legend(loc='best')
plt.xlim((4, 5))
plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV)')

###################################################################################################
#
# Compute and plot instantaneous frequency
# ----------------------------------------
#
# Note that some people apply median filters to this instantaneous frequency
# estimate in order to make it smoother (see e.g. Samaha & Postle, 2015)
#

###################################################################################################

# Compute instaneous frequency from a signal
i_f = freq_by_time(sig, fs, f_range)

###################################################################################################

# Plot example signal
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(times, sig, 'k', label='raw voltage')
plt.legend(loc='best')
plt.ylabel('Voltage (uV)')
plt.xlim((4, 5))
plt.subplot(3, 1, 2)
plt.plot(times, sig_filt_true, 'b', label='beta-filtered voltage')
plt.legend(loc='best')
plt.ylabel('Voltage (uV)')
plt.xlim((4, 5))
plt.subplot(3, 1, 3)
plt.plot(times, i_f, 'r')
plt.legend(loc='best')
plt.xlim((4, 5))
plt.xlabel('Time (s)')
plt.ylabel('Instantaneous\nFrequency (Hz)')
plt.ylim((10, 30))

###################################################################################################
#
# Sphinx settings:
# sphinx_gallery_thumbnail_number = 3
#
