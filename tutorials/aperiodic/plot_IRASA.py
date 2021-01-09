"""
IRASA
=====

Apply the IRASA algorithm to separate periodic and aperiodic activity.

The irregular resampling auto-spectral analysis (IRASA) algorithm is a method for separating
aperiodic (1/f) and oscillatory activity in the frequency domain. This technique involves:

1. Up/Downsampling the signal across a range of increments.
2. Computing the geometric mean spectra for each pair of up/down sampled signals.
3. Estimating the aperiodic component from the median spectrum across the range of increments.
4. Estimating the periodic component from the difference between the aperiodic estimate and the
   original spectrum.

The IRASA algorithm is described in
`Wen & Liu, 2016 <https://doi.org/10.1007/s10548-015-0448-0>`_

This tutorial covers ``neurodsp.aperiodic.irasa``.
"""

###################################################################################################

import numpy as np
from neurodsp.sim import sim_combined
from neurodsp.spectral import compute_spectrum, trim_spectrum
from neurodsp.plts import plot_power_spectra
from neurodsp.aperiodic import compute_irasa, fit_irasa

###################################################################################################
# Simulate Data
# -------------
#
# First, a signal is simulated to contain powerlaw and oscillatory components. The resulting
# spectrum will contain a 1/f slope with a slope of -2 and an oscillatory peak at 10hz.
#

# Simulate
n_seconds = 10
fs = 500
components = dict(sim_oscillation=dict(freq=10), sim_powerlaw=dict(exponent=-2))
sig = sim_combined(n_seconds, fs, components)

# Compute and plot the spectrum
f_range = (1, 40)
freqs, psd = compute_spectrum(sig, fs, nperseg=4*fs)
freqs, psd = trim_spectrum(freqs, psd, f_range)
plot_power_spectra(freqs, psd, title="Original Spectrum")

###################################################################################################
# IRASA
# -----
#
# Next, IRASA is ran on the simulate signal and the results are plotted. The IRASA results are
# thresholded such that regions of the periodic component that are not >= 2 standard deviations
# above the aperiodic component, become apart of the aperiodic component. This allows for a near
# perfect separation of the spectral components.
#

# Fit
freqs, psd_aperiodic, psd_periodic = compute_irasa(sig, fs, f_range=f_range, thresh=2)

# Plot
plot_power_spectra(freqs, [psd_aperiodic, psd_periodic], title="IRASA")

###################################################################################################
#
# Results
# -------
#
# The results are confirmed by ensuring the the original spectrum may be resconstructed from the
# sum of periodic and aperiodic components. A linear fit of the aperiodic component also confirms
# that the slope of the aperiodic is approximately the simulated slope.
#

# The sum of IRASA components is the original signal
psd_irasa = psd_aperiodic + psd_periodic
assert np.equal(psd_irasa, psd).all()

# Fit the aperiodic component of the IRASA results
intercept, exp = fit_irasa(freqs, psd_aperiodic)
print("IRASA Exponent: {exp}".format(exp=exp))
