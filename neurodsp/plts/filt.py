"""Plotting functions for neurodsp.filt."""

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

def plot_frequency_response(fs, b_vals, a_vals=1):
    """Compute frequency response of a filter kernel b with sampling rate fs.

    Parameters
    ----------
    fs : float
        The sampling rate, in Hz.
    b_vals : 1d array
        B values for the filter.
    a_vals : 1d array
        A values for the filter.
    """

    w_vals, h_vals = signal.freqz(b_vals, a_vals)

    # Plot frequency response
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(w_vals * fs / (2. * np.pi), 20 * np.log10(abs(h_vals)), 'k')
    plt.title('Frequency response')
    plt.ylabel('Attenuation (dB)')
    plt.xlabel('Frequency (Hz)')

    # Plot filter kernel, if available
    if isinstance(a_vals, int):

        plt.subplot(1, 2, 2)
        plt.plot(b_vals, 'k')
        plt.title('Kernel')
