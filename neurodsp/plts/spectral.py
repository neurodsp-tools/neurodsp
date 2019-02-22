"""Plotting functions for neurodsp.spectral."""

import numpy as np
import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

def plot_spectral_hist(freqs, power_bins, spect_hist, spectrum_freqs=None, spectrum=None):
    """Plot the spectral histogram.

    Parameters
    ----------
    freqs : 1d array
        Frequencies over which the histogram is calculated.
    power_bins : 1d array
        Power bins within which histogram is aggregated.
    spect_hist : 2d array
        Spectral histogram to be plotted.
    spectrum_freqs : 1d array, optional
        Frequency axis of the power spectrum to be plotted.
    spectrum : 1d array, optional
        Spectrum to be plotted over the histograms.
    """

    # Automatically scale figure height based on number of bins
    plt.figure(figsize=(8, 12 * len(power_bins) / len(freqs)))

    # Plot histogram intensity as image and automatically adjust aspect ratio
    plt.imshow(spect_hist, extent=[freqs[0], freqs[-1], power_bins[0], power_bins[-1]], aspect='auto')
    plt.xlabel('Frequency (Hz)', fontsize=15)
    plt.ylabel('Log10 Power', fontsize=15)
    plt.colorbar(label='Probability')

    # If a PSD is provided, plot over the histogram data
    if spectrum is not None:
        plt.plot(spectrum_freqs[np.logical_and(spectrum_freqs >= freqs[0], spectrum_freqs <= freqs[-1])],
                 np.log10(spectrum[np.logical_and(spectrum_freqs >= freqs[0], spectrum_freqs <= freqs[-1])]),
                 color='w', alpha=0.8)
