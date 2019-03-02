"""
Using NeuroDSP with MNE
=======================

This example explores some example analyses using NeuroDSP, integrated with MNE.
"""

###################################################################################################

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample

from neurodsp.spectral import compute_spectrum, trim_spectrum
from neurodsp.burst import detect_bursts_dual_threshold

###################################################################################################

# Get the data path for the MNE example data
raw_fname = sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

# Load the file of example MNE data
raw = io.read_raw_fif(raw_fname, preload=True, verbose=False)

###################################################################################################

# Select EEG channels from the dataset
raw = raw.pick_types(meg=False, eeg=True, eog=True, exclude='bads')

# Grab the sampling rate from the data
fs = raw.info['sfreq']

###################################################################################################

# Settings
ch_label = 'EEG 058'
t_start = 20000
t_stop = int(t_start + (10 * fs))

###################################################################################################

# Extract an example channel to explore
sig, times = raw.get_data(mne.pick_channels(raw.ch_names, [ch_label]), start=t_start, stop=t_stop, return_times=True)
sig = np.squeeze(sig)

###################################################################################################

# Plot a segment of the extracted time series data
plt.figure(figsize=(16, 3))
plt.plot(times, sig, 'k')

###################################################################################################

# Calculate the power spectrum, using a median welch & extract a frequency range of interest
freqs, powers = compute_spectrum(sig, fs, method='median')
freqs, powers = trim_spectrum(freqs, powers, [3, 30])

###################################################################################################

# Check where the peak power is, in center frequency
peak_cf = freqs[np.argmax(powers)]
print(peak_cf)

###################################################################################################

# Plot the power spectra
plt.figure(figsize=(8, 8))
plt.semilogy(freqs, powers)
plt.plot(freqs[np.argmax(powers)], np.max(powers), '.r', ms=12)

###################################################################################################

# Burst Settings
amp_dual_thresh = (1., 1.5)
f_range = (peak_cf-2, peak_cf+2)

###################################################################################################

# Detect bursts of high amplitude oscillations in the extracted signal
bursting = detect_bursts_dual_threshold(sig, fs, f_range, amp_dual_thresh)

###################################################################################################

# Plot original signal and burst activity
plt.figure(figsize=(16, 3))
plt.plot(times, sig, 'k', label='Raw Data')
plt.plot(times[bursting], sig[bursting], 'r', label='Detected Bursts')
plt.legend(loc='best')

###################################################################################################
