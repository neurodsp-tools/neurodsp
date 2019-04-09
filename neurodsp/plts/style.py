"""Style helpers and utilities for NeuroDSP plots."""

from functools import wraps

import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

PLOT_STYLE_ARGS = ['title', 'xlabel', 'ylabel', 'xlim', 'ylim']

def plot_style(ax, **kwargs):
    """Define plot style."""

    # Set any provided plot arguments
    ax.set(**kwargs)

    # Aesthetics and axis labels
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # If labels were provided, add a legend
    if ax.get_legend_handles_labels()[0]:
        ax.legend(prop={'size': 12}, loc='best')

    # If a title was provided, update the size
    if ax.get_title():
        ax.title.set_size(20)

    plt.tight_layout()


def style_plot(func, *args, **kwargs):
    """Decorator function to apply a plot style function, after plot generation."""

    @wraps(func)
    def decorated(*args, **kwargs):

        style_kwargs = {key : kwargs.pop(key) for key in PLOT_STYLE_ARGS if key in kwargs}

        style_func = kwargs.pop('plot_style', plot_style)
        func(*args, **kwargs)
        style_func(plt.gca(), **style_kwargs)

    return decorated
