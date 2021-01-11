"""
Autocorrelation Measures
========================

Apply autocorrelation measures to neural signals.

This tutorial covers ``neurodsp.aperiodic.autocorr``.
"""

###################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.sim import sim_powerlaw, sim_oscillation
from neurodsp.aperiodic import compute_autocorr

###################################################################################################

###################################################################################################
# Simulate Data
# -------------
#
# A sine wave will be used for the first example to highlight how autocorrelations map to the
# phase of an oscillatory signal. A second signal, containing white noise, will be used to
# highlight the absence of an autocorrelation structure.
#

###################################################################################################

# Simulation settings
n_seconds = 10
fs = 1000
freq = 10
freq_samp = int(fs / freq)

# Simulate signals
sig_osc = sim_oscillation(n_seconds, fs, freq)
sig_wn = sim_powerlaw(n_seconds, fs, exponent=0)

###################################################################################################
# Compute Autocorrelations
# ------------------------
#
# The autocorrelation will be computed between the original signal and sliced signals using various
# lag lengths. The lagged signals are shifted in steps of 1 from 0 to 1000 samples (default). The
# :func:`~.compute_autocorr` function returns ``timepoints``, the original signal sample indices
# where each lagged signal begins, and ``autocorrs``, the correlation coefficient between the
# original signal and each lagged signal.
#

# Compute the autocorrelations using various lag lengths
timepoints_osc, autocorrs_osc = compute_autocorr(sig_osc)
timepoints_wn, autocorrs_wn = compute_autocorr(sig_wn)

###################################################################################################
#
# Result: Sine Wave
# -----------------
#
# The autocorrlations of the sine wave simulation are plotted below. The maximum, minimum,
# decay zero-crossings, rise zero-crossings of the autocorrelation plot correspond to a phase shift
# of 2pi, pi, pi/2, and 3pi/2, respectively.
#

# Define max, min, and zero-crossing autocorrlation locations
max_lag = 1000
phase_2_pi = np.arange(0, max_lag, freq_samp, dtype=int)
phase_pi = np.arange(0, max_lag, freq_samp/2, dtype=int)[1::2]
phase_pi_2 = np.arange(0, max_lag, freq_samp/4, dtype=int)[1::4]
phase_3pi_2 = np.arange(0, max_lag, freq_samp/4, dtype=int)[3::4]

# Plot autocorrelations
figsize=(10, 8)
fig, ax = plt.subplots(figsize=figsize)

ax.plot(timepoints_osc, autocorrs_osc, label="Auto-correlations")

ax.plot(phase_2_pi, autocorrs_osc[phase_2_pi],
        ls="", marker="o", ms=6, label="Phase shift: 2pi")

ax.plot(phase_pi, autocorrs_osc[phase_pi],
        ls="", marker="o", ms=6, label="Phase shift: pi")

ax.plot(phase_pi_2, autocorrs_osc[phase_pi_2],
        ls="", marker="o", ms=6, label="Phase shift: pi/2")

ax.plot(phase_3pi_2, autocorrs_osc[phase_3pi_2],
        ls="", marker="o", ms=6, label="Phase shift: 3pi/2")

ax.set_ylabel("Auto-Correlation Coefficient")
ax.set_xlabel("Lag Length (samples)")
ax.set_title("Auto-Correlation: Sine Wave")
ax.legend(loc="upper right")

###################################################################################################

# Plot phase shifts
fig, ax = plt.subplots(figsize=figsize)

cyc_times = np.arange(0, freq_samp)

# Reference cycle
ax.plot(cyc_times, sig_osc[:freq_samp], label="Reference")

# Phase shift: 2pi
ax.plot(cyc_times, sig_osc[freq_samp:2*freq_samp],
        ls="-", dashes=(4, 4), label="Phase: 2pi")

# Phase shift: pi
ax.plot(cyc_times, sig_osc[int(freq_samp/2):freq_samp+int(freq_samp/2):],
        ls="-", dashes=(4, 4), label="Phase: pi")

# Phase shift: pi/2
ax.plot(cyc_times, sig_osc[int(freq_samp/4):freq_samp+int(freq_samp/4):],
        ls="-", dashes=(4, 4), label="Phase: pi/2")

#Phase shift: 3pi/2
ax.plot(cyc_times, sig_osc[int(3*freq_samp/4):freq_samp+int(3*freq_samp/4):],
        ls="-", dashes=(4, 4), label="Phase: 3pi/2")

ax.set_xlabel("Samples")
ax.set_ylabel("Signal")
ax.set_title("Phase Shifts")
ax.legend(loc="upper right")

###################################################################################################
#
# Result: White Noise
# -------------------
#
# The autocorrelation of a signal containing white noise will have a single peak (dirac) at
# ``timepoints == 0``. The autocorrelation coefficient at all other timepoints will be near zero.
#

fig, ax = plt.subplots(figsize=figsize)
ax.plot(timepoints_wn, autocorrs_wn)
ax.set_ylabel("Auto-Correlation Coefficient")
ax.set_xlabel("Lag Length (samples)")
ax.set_title("Auto-Correlation: White Noise")
