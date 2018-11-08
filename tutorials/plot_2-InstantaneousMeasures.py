"""
Time-frequency analysis
=======================
In this tutorial, we will show how to estimate instantaneous measures of phase,
amplitude, and frequency.
"""
###############################################################################

import numpy as np
from neurodsp.timefrequency import amp_by_time, freq_by_time, phase_by_time
import matplotlib.pyplot as plt

###############################################################################
#
# Load example neural signal
# --------------------------

sig = np.load('./data/sample_data_1.npy')
Fs = 1000
t = np.arange(0, len(sig)/Fs, 1/Fs)
f_range = (13,30)

# Load filtered version of signal
sig_filt_true = np.load('./data/sample_data_1_filt.npy')

# Plot signal
plt.figure(figsize=(12,3))
plt.plot(t, sig, 'k')
plt.xlim((4,5))
plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV)')

###############################################################################
#
# Compute and plot instantaneous phase
# ------------------------------------

pha = phase_by_time(sig, Fs, f_range)

# Plot example signal
plt.figure(figsize=(12,4))
plt.subplot(2,1,1)
plt.plot(t, sig, 'k')
plt.xlim((4,5))
plt.ylabel('Voltage (uV)')
plt.subplot(2,1,2)
plt.plot(t, pha, 'k')
plt.xlim((4,5))
plt.yticks([-np.pi, 0, np.pi], ['$\pi$',0,'$\pi$'])
plt.xlabel('Time (s)')
plt.ylabel('Phase (rad)')

###############################################################################
#
# Compute and plot instantaneous amplitude
# ----------------------------------------

amp = amp_by_time(sig, Fs, f_range)

# Plot example signal
plt.figure(figsize=(12,4))
plt.subplot(2,1,1)
plt.plot(t, sig, 'k', label='raw voltage')
plt.plot(t, amp, 'r', label='amplitude')
plt.legend(loc='best')
plt.xlim((4,5))
plt.ylabel('Voltage (uV)')
plt.subplot(2,1,2)
plt.plot(t, sig_filt_true, 'b', label='beta-filtered voltage')
plt.plot(t, amp, 'r', label='amplitude')
plt.legend(loc='best')
plt.xlim((4,5))
plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV)')

###############################################################################
#
# Compute and plot instantaneous frequency
# ----------------------------------------
#
# Note that some people apply median filters to this instantaneous frequency
# estimate in order to make it smoother (see e.g. Samaha & Postle, 2015)

i_f = freq_by_time(sig, Fs, f_range)

# Plot example signal
plt.figure(figsize=(12,6))
plt.subplot(3,1,1)
plt.plot(t, sig, 'k', label='raw voltage')
plt.legend(loc='best')
plt.ylabel('Voltage (uV)')
plt.xlim((4,5))
plt.subplot(3,1,2)
plt.plot(t, sig_filt_true, 'b', label='beta-filtered voltage')
plt.legend(loc='best')
plt.ylabel('Voltage (uV)')
plt.xlim((4,5))
plt.subplot(3,1,3)
plt.plot(t, i_f, 'r')
plt.legend(loc='best')
plt.xlim((4,5))
plt.xlabel('Time (s)')
plt.ylabel('Instantaneous\nFrequency (Hz)')
plt.ylim((10,30))
