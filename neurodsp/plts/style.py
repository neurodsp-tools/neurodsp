"""Style helpers and utilities for plots."""

from itertools import cycle
from functools import wraps

import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

PLOT_STYLE_ARGS = ['title', 'xlabel', 'ylabel', 'xlim', 'ylim']
LINE_STYLE_ARGS = ['alpha', 'lw', 'linewidth', 'ls', 'linestyle']
STYLE_ARGS = PLOT_STYLE_ARGS + LINE_STYLE_ARGS

def plot_style(ax, **kwargs):
    """Define plot style."""

    # Apply any provided plot style arguments
    plot_kwargs = {key : val for key, val in kwargs.items() if key in PLOT_STYLE_ARGS}
    ax.set(**plot_kwargs)

    # Apply any provided line style arguments
    line_kwargs = {key : val for key, val in kwargs.items() if key in LINE_STYLE_ARGS}
    for style, value in line_kwargs.items():

        # Values should be either a single value, for all lines, or a list, of a value per line
        #   This line checks type, and makes a cycle-able / loop-able object out of the values
        values = cycle([value] if isinstance(value, (int, float, str)) else value)
        for idx, line in enumerate(ax.lines):
            line.set(**{style : next(values)})

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

        # Grab a custom style function, if provided, and grab any provided style arguments
        style_func = kwargs.pop('plot_style', plot_style)
        style_kwargs = {key : kwargs.pop(key) for key in STYLE_ARGS if key in kwargs}

        # Create the plot
        func(*args, **kwargs)

        # Get plot axis, if a specific one was provided, or just grab current and apply style
        cur_ax = kwargs['ax'] if 'ax' in kwargs and kwargs['ax'] is not None else plt.gca()
        style_func(cur_ax, **style_kwargs)

    return decorated
