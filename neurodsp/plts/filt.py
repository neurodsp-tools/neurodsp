"""Plotting functions for neurodsp.filt."""

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

def plot_filter_properties(f_db, db, b_vals):
    """Plot filter properties, including frequency response and filter kernel.

    Parameters
    ----------
    f_db : 1d array
        Frequency vector corresponding to attenuation decibels, in Hz.
    db : 1d array
        Degree of attenuation for each frequency specified in f_db, in dB.
    b_vals : 1d array
        B values for the filter.
    """

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    plot_frequency_response(f_db, db, ax=ax[0])
    plot_filter_kernel(b_vals, ax=ax[1])


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

    if not ax:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.plot(f_db, db, 'k')

    ax.set_title('Frequency response')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Attenuation (dB)')


def plot_filter_kernel(b_vals, ax=None):
    """Plot the kernel of a filter.

    Parameters
    ----------
    b_vals : 1d array
        B values for the filter.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    """

    if not ax:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.plot(b_vals, 'k')

    ax.set_title('Kernel')
    ax.set_xlabel('LABEL')
    ax.set_ylabel('LABEL')
