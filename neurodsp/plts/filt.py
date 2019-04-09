"""Plotting functions for neurodsp.filt."""

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.plts.style import style_plot
from neurodsp.plts.utils import check_ax, savefig

###################################################################################################
###################################################################################################

@savefig
def plot_filter_properties(f_db, db, fs, impulse_response):
    """Plot filter properties, including frequency response and filter kernel.

    Parameters
    ----------
    f_db : 1d array
        Frequency vector corresponding to attenuation decibels, in Hz.
    db : 1d array
        Degree of attenuation for each frequency specified in f_db, in dB.
    impulse_response : 1d array
        The impulse response of a filter. For an FIR filter, these are the filter coefficients.
    """

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    plot_frequency_response(f_db, db, ax=ax[0])
    plot_impulse_response(fs, impulse_response, ax=ax[1])


@savefig
@style_plot
def plot_frequency_response(f_db, db, ax=None):
    """Plot the frequency response of a filter.

    Parameters
    ----------
    f_db : 1d array
        Frequency vector corresponding to attenuation decibels, in Hz.
    db : 1d array
        Degree of attenuation for each frequency specified in f_db, in dB.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    """

    ax = check_ax(ax, (5, 5))

    ax.plot(f_db, db, 'k')

    ax.set_title('Frequency response')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Attenuation (dB)')


@savefig
@style_plot
def plot_impulse_response(fs, impulse_response, ax=None):
    """Plot the impulse response of a filter.

    Parameters
    ----------
    impulse_response : 1d array
        The impulse response of a filter.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    """

    ax = check_ax(ax, (5, 5))

    # Create a samples vector, center to zero, and convert to time
    samples = np.arange(len(impulse_response))
    samples = samples - (len(samples) - 1) / 2
    time = samples / fs

    ax.plot(time, impulse_response, 'k')

    ax.set_title('Kernel')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Response')
