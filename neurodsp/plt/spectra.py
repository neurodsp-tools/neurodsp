"""Plotting functions for neurodsp."""

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

def plot_frequency_response(fs, b_vals, a_vals=1):
    """Compute frequency response of a filter kernel b with sampling rate fs"""

    w_vals, h_vals = signal.freqz(b_vals, a_vals)

    # Plot frequency response
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(w_vals * fs / (2. * np.pi), 20 * np.log10(abs(h_vals)), 'k')
    plt.title('Frequency response')
    plt.ylabel('Attenuation (dB)')
    plt.xlabel('Frequency (Hz)')

    if isinstance(a_vals, int):

        # Plot filter kernel
        plt.subplot(1, 2, 2)
        plt.plot(b_vals, 'k')
        plt.title('Kernel')

    plt.show()


def plot_spectral_hist(freq, power_bins, spect_hist, spectrum_freqs=None, spectrum=None):
    """Plot the spectral histogram.

    Parameters
    ----------
    freq : array_like, 1d
        Frequencies over which the histogram is calculated.
    power_bins : array_like, 1d
        Power bins within which histogram is aggregated.
    spect_hist : ndarray, 2d
        Spectral histogram to be plotted.
    spectrum_freqs : array_like, 1d, optional
        Frequency axis of the PSD to be plotted.
    spectrum : array_like, 1d, optional
        spectrum to be plotted over the histograms.
    """

    # automatically scale figure height based on number of bins
    plt.figure(figsize=(8, 12 * len(power_bins) / len(freq)))

    # plot histogram intensity as image and automatically adjust aspect ratio
    plt.imshow(spect_hist, extent=[freq[0], freq[-1], power_bins[0], power_bins[-1]], aspect='auto')
    plt.xlabel('Frequency (Hz)', fontsize=15)
    plt.ylabel('Log10 Power', fontsize=15)
    plt.colorbar(label='Probability')

    if spectrum is not None:
        # if a PSD is provided, plot over the histogram data
        plt.plot(spectrum_freqs[np.logical_and(spectrum_freqs >= freq[0], spectrum_freqs <= freq[-1])], np.log10(
            spectrum[np.logical_and(spectrum_freqs >= freq[0], spectrum_freqs <= freq[-1])]), color='w', alpha=0.8)
