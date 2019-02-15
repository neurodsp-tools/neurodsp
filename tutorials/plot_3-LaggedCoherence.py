"""
Lagged Coherence
================

This notebook shows how to use the neurodsp module to compute lagged coherence.
For more details, see Fransen et al., 2015, Neuroimage.
"""

###################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.laggedcoherence import lagged_coherence

###################################################################################################

# Set the random seed, for consistency simulating data
np.random.seed(0)

###################################################################################################
#
# Simulate signal with oscillatory bursts
# ---------------------------------------
#

###################################################################################################

# Parameters for simulated signal
N = 5000
Fs = 1000
burst_freq = 10
burst_starts = [0, 3000]
burst_seconds = 1
burst_samples = burst_seconds*Fs

###################################################################################################

# Design burst kernel
burst_kernel_t = np.arange(0, burst_seconds, 1/Fs)
burst_kernel = 2*np.sin(burst_kernel_t*2*np.pi*burst_freq)

# Generate random signal with bursts
t = np.arange(0, N/Fs, 1/Fs)
x = np.random.randn(N)
for i in burst_starts:
    x[i:i+burst_samples] += burst_kernel

###################################################################################################

# Plot example signal
plt.figure(figsize=(12,3))
plt.plot(t, x, 'k')

###################################################################################################
#
# Compute lagged coherence for an alpha oscillation
# -------------------------------------------------

f_range = (8, 12)
lag_coh_alpha = lagged_coherence(x, f_range, Fs)
print('Lagged coherence = ', lag_coh_alpha)

###################################################################################################
#
# Compute lagged coherence across the frequency spectrum
# ------------------------------------------------------
#
# Notice that lagged coherence peaks around 10Hz (the frequency of our
# oscillator), but it is not very specific to that frequency.

lag_coh_by_f, f = lagged_coherence(x, (1, 40), Fs, return_spectrum=True)

# Visualize lagged coherence as a function of frequency
plt.figure(figsize=(6,3))
plt.plot(f, lag_coh_by_f, 'k.-')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Lagged coherence')
plt.tight_layout()

###################################################################################################
#
# Compute lagged coherence for time segments with and without burst
# -----------------------------------------------------------------
#
# Note that lagged coherence is greater when analyzing a neural signal that
# has a burst in the frequency range of interest, compared to a signal that
# does not have an oscillator.

samp_burst = np.arange(1000)
samp_noburst = np.arange(1000, 2000)

lag_coh_burst = lagged_coherence(x[samp_burst], f_range, Fs)
lag_coh_noburst = lagged_coherence(x[samp_noburst], f_range, Fs)
print('Lagged coherence, bursting = ', lag_coh_burst)
print('Lagged coherence, not bursting = ', lag_coh_noburst)

###################################################################################################
#
# Compute lagged coherence of an example neural signal
# ----------------------------------------------------

# Load signal
x = np.load('./data/sample_data_1.npy')
x_filt_true = np.load('./data/sample_data_1_filt.npy')
Fs = 1000
t = np.arange(0, len(x)/Fs, 1/Fs)
f_range = (13,30)

###################################################################################################

# Plot example signal
plt.figure(figsize=(12,3))
plt.plot(t, x, 'k')
plt.xlim((0,5))
plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV)')
plt.tight_layout()

###################################################################################################

f_range = (13, 30)
lag_coh_beta = lagged_coherence(x, f_range, Fs)
print('Lagged coherence = ', lag_coh_beta)
