"""
Spectral Domain Analysis: Variance
==================================

Apply spectral domain analyses, calculating variance measures.

This tutorial primarily covers :mod:`neurodsp.spectral.variance`.
"""

###################################################################################################
#
# Overview
# --------
#
# This tutorial covers computing and displaying a spectral histogram, and
# computing the spectral coefficient of variation (SCV).
#

###################################################################################################

import numpy as np

from neurodsp.utils import create_times
from neurodsp.plts.time_series import plot_time_series
from neurodsp.plts.spectral import plot_spectral_hist
from neurodsp.plts.spectral import plot_scv, plot_scv_rs_lines, plot_scv_rs_matrix

from neurodsp import spectral

###################################################################################################
#
# Load example neural signal
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First, we load the sample data, which is a segment of rat hippocampal LFP
# taken from the publicly available neuro database CRCNS (hc2).
#
# Relevant publication: Mizuseki et al, 2012, Nature Neuro
#

###################################################################################################

# Load example data signal
sig = np.load('../data/sample_data_2.npy')
fs = 1000

# Plot the loaded signal
times = create_times(len(sig)/fs, fs)
plot_time_series(times, sig, xlim=[0, 3])

###################################################################################################
#
# Plotting the data, we observe a strong theta oscillation (~6-8 Hz)
#

###################################################################################################
#
# Spectral histogram
# ------------------
#
# The PSD is an estimate of the central tendency (mean/median) of the signal's power
# at each frequency, with the assumption that the signal is relatively stationary and
# that the variance around the mean comes from various forms of noise.
#
# However, in physiological data, we may be interested in visualizing the distribution of
# power values around the mean at each frequency, as estimated in sequential slices of
# short-time Fourier transform (STFT), since it may reveal non-stationarities in the data
# or particular frequencies that are not like the rest. Here, we simply bin the log-power
# values across time, in a histogram, to observe the noise distribution at each frequency.
#

###################################################################################################

# Calculate the spectral histogram
freqs, bins, spect_hist = spectral.compute_spectral_hist(sig, fs, nbins=50, f_range=(0, 80),
                                                         cut_pct=(0.1, 99.9))

# Calculate a power spectrum, with median Welch
freq_med, psd_med = spectral.compute_spectrum(sig, fs, method='welch',
                                              avg_type='median', nperseg=fs*2)

# Plot the spectral histogram
plot_spectral_hist(freqs, bins, spect_hist, freq_med, psd_med)

###################################################################################################
#
# Notice in the below plot that not only is theta power higher overall (shifted up),
# it also has lower variance around its mean.
#

###################################################################################################
#
# Spectral Coefficient of Variation (SCV)
# ---------------------------------------
#
# As noted above, the range of log-power values in the theta frequency range is lower
# compared to other frequencies, while that of 30-100Hz appear to be quite constant
# across the entire frequency axis (homoscedasticity).
#
# To quantify that, we compute the coefficient of variation (standard deviation/mean) as a
# normalized estimate of variance.
#

###################################################################################################

# Calculate SCV
freqs, scv = spectral.compute_scv(sig, fs, nperseg=int(fs), noverlap=0)

# Plot the SCV
plot_scv(freqs, scv)

###################################################################################################
#
# As shown above, SCV calculated from the entire segment of data is quite noise due to the
# single estimate of mean and standard deviation. To overcome this, we can compute a
# bootstrap-resampled estimate of SCV, by randomly drawing slices from the non-overlapping
# spectrogram and taking their average.
#

###################################################################################################

# Calculate SCV with the resampling method
freqs, t_inds, scv_rs = spectral.compute_scv_rs(sig, fs, nperseg=fs, method='bootstrap',
                                                rs_params=(20, 200))

# Plot the SCV, from the resampling method
plot_scv_rs_lines(freqs, scv_rs)

###################################################################################################
#
# Another way to compute the resampled SCV is via a sliding window approach, essentially
# smoothing over consecutive slices of the spectrogram to compute the mean and standard
# deviation estimates.
#

###################################################################################################

# Calculate SCV with the resampling method
freqs, t_inds, scv_rs = spectral.compute_scv_rs(sig, fs, method='rolling', rs_params=(10, 2))

# Plot the SCV, from the resampling method
plot_scv_rs_matrix(freqs, t_inds, scv_rs)

###################################################################################################
#
# In the plot below, we see that the theta band (~7Hz) consistently has CV of less
# than 1 (negative in log10).
#

###################################################################################################
#
# Sphinx settings:
# sphinx_gallery_thumbnail_number = 4
#
