"""Plotting functions for neurodsp."""

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

def plot_slope_fit(freq, psd, logf, logpsd, slope, offset):
    """Plot slope fit of a power spectrum."""

    plt.figure(figsize=(5, 5))

    plt.plot(np.log10(freq), np.log10(psd), label='Whole PSD')
    plt.plot(logf, logpsd, '-o', label='Fitted PSD', alpha=0.4)
    plt.plot(logf, logf * slope + offset, '-k', label='Fit Line', lw=3)

    plt.legend()

    plt.xlabel('Log10 Frequency (Hz)', fontsize=15)
    plt.ylabel('Log10 Power (V^2/Hz)', fontsize=15)


def plot_frequency_response(Fs, b, a=1):
    """Compute frequency response of a filter kernel b with sampling rate Fs"""

    w, h = signal.freqz(b, a)

    # Plot frequency response
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(w * Fs / (2. * np.pi), 20 * np.log10(abs(h)), 'k')
    plt.title('Frequency response')
    plt.ylabel('Attenuation (dB)')
    plt.xlabel('Frequency (Hz)')

    if isinstance(a, int):

        # Plot filter kernel
        plt.subplot(1, 2, 2)
        plt.plot(b, 'k')
        plt.title('Kernel')

    plt.show()


def plot_spectral_hist(freq, power_bins, spect_hist, psd_freq=None, psd=None):
    """Plot the spectral histogram.

    Parameters
    ----------
    freq : array_like, 1d
        Frequencies over which the histogram is calculated.
    power_bins : array_like, 1d
        Power bins within which histogram is aggregated.
    spect_hist : ndarray, 2d
        Spectral histogram to be plotted.
    psd_freq : array_like, 1d, optional
        Frequency axis of the PSD to be plotted.
    psd : array_like, 1d, optional
        PSD to be plotted over the histograms.
    """

    # automatically scale figure height based on number of bins
    plt.figure(figsize=(8, 12 * len(power_bins) / len(freq)))

    # plot histogram intensity as image and automatically adjust aspect ratio
    plt.imshow(spect_hist, extent=[freq[0], freq[-1], power_bins[0], power_bins[-1]], aspect='auto')
    plt.xlabel('Frequency (Hz)', fontsize=15)
    plt.ylabel('Log10 Power', fontsize=15)
    plt.colorbar(label='Probability')

    if psd is not None:
        # if a PSD is provided, plot over the histogram data
        plt.plot(psd_freq[np.logical_and(psd_freq >= freq[0], psd_freq <= freq[-1])], np.log10(
            psd[np.logical_and(psd_freq >= freq[0], psd_freq <= freq[-1])]), color='w', alpha=0.8)
