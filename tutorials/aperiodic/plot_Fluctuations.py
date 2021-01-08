"""
Fluctuation analyses
====================

Apply fluctuation analyses, such as DFA to neural signals.

This tutorial covers ``neurodsp.aperiodic.dfa``.
"""

###################################################################################################

import matplotlib.pyplot as plt

from neurodsp.sim import sim_powerlaw
from neurodsp.aperiodic import compute_fluctuations

###################################################################################################

###################################################################################################
# Simulate Data
# -------------
#
# Detrended fluctuation analysis calculates flucations for various window sizes. The DFA exponent
# is the slope of the linear fit between windows sizes and flucuations. In this tutorial, we will
# simulate two signals (white noise and 1/f) to compare DFA exponents. These exponents represent:
#
#   - :math:`0 < \alpha < 0.5` : anti-correlated
#   - :math:`\alpha = 0.5` : white noise
#   - :math:`0.5 < \alpha < 1` : correlated (stationary)
#   - :math:`\alpha = 1.0` : 1/f
#   - :math:`1 < \alpha < 2` : correlated (non-stationary)
#

###################################################################################################

# Sim settings
n_seconds = 10
fs = 500

# White noise
sig_wn = sim_powerlaw(n_seconds, fs, exponent=0)

# Power-law
sig_pl = sim_powerlaw(n_seconds, fs, exponent=-1)

###################################################################################################
# Running DFA
# -----------
#
# DFA involves:
#
#   1. Removing the mean of a signal (detrend)
#   2. Computing the cumulative sum of the signal
#   3. Splitting the signal in equal-sized windows
#   4. Fitting a polynomial across the windows
#   5. Calculate the mean squared residual (flucuation) of the fit
#
# Steps 3-5 are repeated for various window sizes.

###################################################################################################

ts_wn, flucs_wn, exp_wn = compute_fluctuations(sig_wn, fs, n_scales=10, min_scale=0.01,
                                               max_scale=1.0, deg=1, method='dfa')

ts_pl, flucs_pl, exp_pl = compute_fluctuations(sig_pl, fs, n_scales=10, min_scale=0.01,
                                               max_scale=1.0, deg=1, method='dfa')

###################################################################################################
# Results
# -------
#
# Below, flucuations are plotted across window sizes for both signals in log-log space. The DFA
# exponent (or slope in log-log space) for the white noise signal is ~=0.5 and the exponent for the
# powerlaw signal is ~=1.

###################################################################################################

fig = plt.figure(figsize=(5, 5))

wn_label = "White Noise DFA Exponent = {exp}".format(exp=round(exp_wn, 3))
pl_label = "Power Law DFA Exponent = {exp}".format(exp=round(exp_pl, 3))

plt.loglog(ts_wn, flucs_wn, label=wn_label)
plt.loglog(ts_pl, flucs_pl, label=pl_label)

plt.title("Flucuations Across Window Sizes")
plt.xlabel("Time Scales")
plt.ylabel("Fluctuations")
plt.legend()
