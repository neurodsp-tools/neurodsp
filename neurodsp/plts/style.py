"""Functions and utilities to apply aesthetic styling to plots."""

from itertools import cycle
from functools import wraps

import matplotlib.pyplot as plt

from neurodsp.plts.settings import PLOT_STYLE_ARGS, LINE_STYLE_ARGS, STYLE_ARGS
from neurodsp.plts.settings import (LABEL_SIZE, LEGEND_SIZE, LEGEND_LOC,
                                    TICK_LABELSIZE, TITLE_FONTSIZE)

###################################################################################################
###################################################################################################

def plot_style(ax, **kwargs):
    """Apply plot style to a figure axis.

    Parameters
    ----------
    ax : matplotlib.Axes
        Figure axes to apply style to.
    **kwargs
        Keyword arguments that define plot style to apply.
    """

    # Apply any provided plot style arguments
    plot_kwargs = {key : val for key, val in kwargs.items() if key in PLOT_STYLE_ARGS}
    ax.set(**plot_kwargs)

    # Apply any provided line style arguments
    line_kwargs = {key : val for key, val in kwargs.items() if key in LINE_STYLE_ARGS}
    for style, value in line_kwargs.items():

        # Values should be either a single value, for all lines, or a list, of a value per line
        #   This line checks type, and makes a cycle-able / loop-able object out of the values
        values = cycle([value] if isinstance(value, (int, float, str)) else value)
        for line in ax.lines:
            line.set(**{style : next(values)})

    # If a title was provided, update the size
    if ax.get_title():
        ax.title.set_size(kwargs.pop('title_fontsize', TITLE_FONTSIZE))

    # Settings for the axis labels
    label_size = kwargs.pop('label_size', LABEL_SIZE)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)

    # Settings for the axis ticks
    ax.tick_params(axis='both', which='major',
                   labelsize=kwargs.pop('tick_labelsize', TICK_LABELSIZE))

    # If labels were provided, add a legend
    if ax.get_legend_handles_labels()[0]:
        ax.legend(prop={'size': kwargs.pop('legend_size', LEGEND_SIZE)},
                  loc=kwargs.pop('legend_loc', LEGEND_LOC))

    plt.tight_layout()


def style_plot(func, *args, **kwargs):
    """Decorator function to apply a plot style function, after plot generation.

    Parameters
    ----------
    func : callable
        The plotting function for creating a plot.
    *args, **kwargs
        Arguments & keyword arguments.
        These should include any arguments for the plot, and those for applying plot style.

    Notes
    -----
    This is a decorate, for plot, functions that functions roughly as:

    - catching all inputs that relate to plot style
    - create a plot, using the passed in plotting function & passing in all non-style arguments
    - passing the style related arguments into a `plot_style` function

    This function itself does not apply create any plots or apply any styling itself.

    By default, this function applies styling with the `plot_style` function. Custom
    functions for applying style can be passed in using `plot_style` as a keyword argument.
    """

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
