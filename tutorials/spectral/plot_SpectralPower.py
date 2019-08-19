"""
Spectral domain analysis
========================

Apply spectral domain analyses.

This tutorial primarily covers :mod:`neurodsp.spectral`.
"""

###################################################################################################
#
# Overview
# --------
#
# This tutorial covers:
#
# - Computing the power spectral density (PSD)
# - Computing and displaying the spectral histogram
# - Computing the spectral coefficient of variation (SCV) and the resampled estimates
#
# Detailed explanations for each of the above concept are provided below as we come across them.
#

###################################################################################################

import numpy as np
from scipy import io

from neurodsp import spectral
from neurodsp.utils import create_times
from neurodsp.plts.time_series import plot_time_series
from neurodsp.plts.spectral import *

###################################################################################################

# First, we load the sample data, which is a segment of rat hippocampal LFP
# taken from the publicly available neuro database CRCNS (hc2).
# Relevant publication: Mizuseki et al, 2012, Nature Neuro
data = io.loadmat('../data/sample_data_2.mat', squeeze_me=True)
sig = data['x']
fs = data['fs']

times = create_times(len(sig)/fs, fs)
plot_time_series(times, sig, xlim=[0, 3])

# Plotting the data, we observe a strong theta oscillation (~6-8 Hz)

###################################################################################################
# Computing the power spectral density (PSD)
# ==========================================
#
# A PSD is a frequency domain representation of a time series.
#
# Using the Fourier transform, the signal is split into orthogonal
# components of different frequency.
#
# In this toolbox, there are a few ways of estimating the PSD:
#
# - 'mean' : same as Welch's method; mean over spectrogram of data.
#   This is the most straightforward and "standard" way of computing the PSD.
# - 'median' : similar to Welch's method; median (instead of mean) over
#   spectrogram of data. This diminishes the effect of outlier power values
#   from signal artifacts.
# - 'medfilt' : median filtered of FFT magnitude squared.
#

###################################################################################################

# Mean of spectrogram (Welch)
freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch',
                                                avg_type='mean', nperseg=fs*2)

# Median of spectrogram ("median Welch")
freq_med, psd_med = spectral.compute_spectrum(sig, fs, method='welch',
                                              avg_type='median', nperseg=fs*2)

# Median filtered spectrum
freq_mf, psd_mf = spectral.compute_spectrum(sig, fs, method='medfilt')

###################################################################################################

# Plot the power spectra
plot_power_spectra([freq_mean[:200], freq_med[:200], freq_mf[100:10000]],
                   [psd_mean[:200], psd_med[:200], psd_mf[100:10000]],
                   ['Welch', 'Median Welch', 'Median Filter FFT'])

###################################################################################################
# Aside: Fitting 1/f and oscillations in power spectra
# ----------------------------------------------------
#
# You might notice in the above power spectra that there are regions of band-specific power,
# reflecting oscillations, as well as regions that show linear decrease when plotted in
# log-log scale, in particular the frequency region between 30-100Hz. As well as a large body of
# work investigating oscillations, this "1/f-like" aperiodic components has been found to
# correlate with aging, memory, and cognitive control in several previous publications,
# and has been proposed to reflect the physiological  balance between excitation and inhibition.
#
# See for example: Voytek et al., JNeuro, 2015; Podvalny et al.,
# JNeurophysiol, 2015; Gao et al., NeuroImage, 2017; Lombardi et al., Chaos, 2017
#
# If you are interesting in measuring these periodic (oscillatory) and aperiodic
# (1/f like) components from power spectra, check out the 'fitting oscillations &
# one-over f' `toolbox <https://fooof-tools.github.io/fooof/>`_.
#

###################################################################################################
# Spectral histogram
# ------------------
#
# The PSD is an estimate of the central tendency (mean/median) of the
# signal's power at each frequency, with the assumption that the signal
# is relatively stationary and that the variance around the mean comes
# from various forms of noise.
#
# However, in physiological data, we may be interested in visualizing the
# distribution of power values around the mean at each frequency, as estimated
# in sequential slices of short-time Fourier transform (STFT), since it may
# reveal non-stationarities in the data or particular frequencies that are
# not like the rest. Here, we simply bin the log-power values across time,
# in a histogram, to observe the noise distribution at each frequency.
#

###################################################################################################

# Calculate the spectral histogram
freqs, bins, spect_hist = spectral.compute_spectral_hist(sig, fs, nbins=50, f_range=(0, 80),
                                                         cut_pct=(0.1, 99.9))

# Plot the spectral histogram
plot_spectral_hist(freqs, bins, spect_hist, freq_med, psd_med)

###################################################################################################
#
# Notice in the below plot that not only is theta power higher overall (shifted up),
# it also has lower variance around its mean.
#

###################################################################################################
# Spectral Coefficient of Variation
# =================================
#
# As noted above, the range of log-power values in the theta frequency range
# is lower compared to other frequencies, while that of 30-100Hz appear to be
# quite constant across the entire frequency axis (homoscedasticity). To quantify
# that, we compute the coefficient of variation (standard deviation/mean) as a
# normalized estimate of variance.
#

###################################################################################################

# Calculate SCV
freqs, scv = spectral.compute_scv(sig, fs, nperseg=int(fs), noverlap=0)

# Plot the SCV
plot_scv(freqs, scv)

###################################################################################################
#
# As shown above, SCV calculated from the entire segment of data is quite
# noise due to the single estimate of mean and standard deviation. To overcome
# this, we can compute a bootstrap-resampled estimate of SCV, by randomly drawing
# slices from the non-overlapping spectrogram and taking their average.
#

###################################################################################################

# Calculate SCV with the resampling method
freqs, t_inds, scv_rs = spectral.compute_scv_rs(sig, fs, nperseg=fs, method='bootstrap',
                                                rs_params=(20, 200))

# Plot the SCV, from the resampling method
plot_scv_rs_lines(freqs, scv_rs)

###################################################################################################
#
# Another way to compute the resampled SCV is via a sliding window approach,
# essentially smoothing over consecutive slices of the spectrogram to compute
# the mean and std estimates.
#

###################################################################################################

# Calculate SCV with the resampling method
freqs, t_inds, scv_rs = spectral.compute_scv_rs(sig, fs, method='rolling', rs_params=(10, 2))

# Plot the SCV, from the resampling method
plot_scv_rs_matrix(freqs, t_inds, scv_rs)

###################################################################################################
#
# In the plot below, we see that the theta band (~7Hz) consistently has CV
# of less than 1 (negative in log10).
#

###################################################################################################
# Spectral Rotation
# =================
#
# Another included utility function is spectral rotation, which rotates
# the power spectrum about a given axis frequency, by an amount indicated
# by the exponent (negative is clockwise, positive is counterclockwise).
#
# This function is mostly useful for investigating the effect of rotating
# the spectrum in frequency domain on the time domain signal. Effectively,
# this performs a very specific type of filtering with an ultra long filter kernel.
#
# For complete details, see the **Simulation** tutorial.
#

###################################################################################################

psd_rot = spectral.rotate_powerlaw(freq_med, psd_med, delta_exponent=-1, f_rotation=35)

plot_power_spectra([freq_med[:200], freq_med[:200]],
                   [psd_med[:200], psd_rot[:200]],
                   ['Original', 'Rotated'])

###################################################################################################
#
# Sphinx settings:
# sphinx_gallery_thumbnail_number = 2
#
