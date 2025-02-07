"""Functions and utilities to apply aesthetic styling to plots."""

import warnings
from itertools import cycle
from functools import wraps

import matplotlib.pyplot as plt

from neurodsp.plts.settings import (AXIS_STYLE_ARGS, LINE_STYLE_ARGS, COLLECTION_STYLE_ARGS,
                                    CUSTOM_STYLE_ARGS, STYLE_ARGS, TICK_LABELSIZE, TITLE_FONTSIZE,
                                    LABEL_SIZE, LEGEND_SIZE, LEGEND_LOC)

###################################################################################################
###################################################################################################

def check_style_options():
    """Check the list of valid style arguments that can be passed into plot functions."""

    print('Valid style arguments:')
    for label, options in zip(['Axis', 'Line', 'Collection', 'Custom'],
                              [AXIS_STYLE_ARGS, LINE_STYLE_ARGS,
                               COLLECTION_STYLE_ARGS, CUSTOM_STYLE_ARGS]):
        print('    {:10s}    {}'.format(label, ', '.join(options)))


def apply_axis_style(ax, style_args=AXIS_STYLE_ARGS, **kwargs):
    """Apply axis plot style.

    Parameters
    ----------
    ax : matplotlib.Axes
        Figure axes to apply style to.
    style_args : list of str
        A list of arguments to be sub-selected from `kwargs` and applied as axis styling.
    **kwargs
        Keyword arguments that define plot style to apply.
    """

    axis_kwargs = {key : val for key, val in kwargs.items() if key in style_args}

    # Special case: catch and apply minorticks being set to True or False
    mtick_dict = {True : 'minorticks_on', False : 'minorticks_off'}
    if 'minorticks' in axis_kwargs:
        getattr(ax, mtick_dict[axis_kwargs.pop('minorticks')])()

    # Apply any provided axis style arguments
    ax.set(**axis_kwargs)


def apply_line_style(ax, style_args=LINE_STYLE_ARGS, **kwargs):
    """Apply line plot style.

    Parameters
    ----------
    ax : matplotlib.Axes
        Figure axes to apply style to.
    style_args : list of str
        A list of arguments to be sub-selected from `kwargs` and applied as line styling.
    **kwargs
        Keyword arguments that define line style to apply.
    """

    # Check how many lines are from the current plot call, to apply style to
    #   If available, this indicates the apply styling to the last 'n' lines
    n_lines_apply = kwargs.pop('n_lines_apply', 0)

    # Get the line related styling arguments from the keyword arguments
    line_kwargs = {key : val for key, val in kwargs.items() if key in style_args}

    # Apply any provided line style arguments
    for style, value in line_kwargs.items():

        # Values should be either a single value, for all lines, or a list, of a value per line
        #   This line checks type, and makes a cycle-able / loop-able object out of the values
        values = cycle([value] if isinstance(value, (int, float, str)) else value)
        for line in ax.lines[-n_lines_apply:]:
            line.set(**{style : next(values)})


def apply_collection_style(ax, style_args=COLLECTION_STYLE_ARGS, **kwargs):
    """Apply collection plot style.

    Parameters
    ----------
    ax : matplotlib.Axes
        Figure axes to apply style to.
    style_args : list of str
        A list of arguments to be sub-selected from `kwargs` and applied as collection styling.
    **kwargs
        Keyword arguments that define collection style to apply.
    """

    # Get the collection related styling arguments from the keyword arguments
    collection_kwargs = {key : val for key, val in kwargs.items() if key in style_args}

    # Apply any provided collection style arguments
    for collection in ax.collections:
        collection.set(**collection_kwargs)


def apply_custom_style(ax, **kwargs):
    """Apply custom plot style.

    Parameters
    ----------
    ax : matplotlib.Axes
        Figure axes to apply style to.
    **kwargs
        Keyword arguments that define custom style to apply.
    """

    # If a title was provided, update the size
    if ax.get_title():
        ax.title.set_size(kwargs.pop('title_fontsize', TITLE_FONTSIZE))

    # Settings for the axis labels, including checking & setting for 3D axis
    label_size = kwargs.pop('label_size', LABEL_SIZE)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    if hasattr(ax, 'zaxis'):
        ax.zaxis.label.set_size(label_size)

    # Settings for the axis ticks
    ax.tick_params(axis='both', which='major',
                   labelsize=kwargs.pop('tick_labelsize', TICK_LABELSIZE))

    # If labels were provided, add a legend
    if ax.get_legend_handles_labels()[0]:
        ax.legend(prop={'size': kwargs.pop('legend_size', LEGEND_SIZE)},
                  loc=kwargs.pop('legend_loc', LEGEND_LOC))

    if kwargs.pop('tight_layout', True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()


def plot_style(ax, axis_styler=apply_axis_style, line_styler=apply_line_style,
               collection_styler=apply_collection_style, custom_styler=apply_custom_style,
               **kwargs):
    """Apply plot style to a figure axis.

    Parameters
    ----------
    ax : matplotlib.Axes
        Figure axes to apply style to.
    axis_styler, line_styler, collection_styler, custom_styler : callable, optional
        Functions to apply style to aspects of the plot.
    **kwargs
        Keyword arguments that define style to apply.

    Notes
    -----
    This function wraps sub-functions which apply style to different plot elements.
    Each of these sub-functions can be replaced by passing in a replacement callable.
    """

    axis_styler(ax, **kwargs) if axis_styler is not None else None
    line_styler(ax, **kwargs) if line_styler is not None else None
    collection_styler(ax, **kwargs) if collection_styler is not None else None
    custom_styler(ax, **kwargs) if custom_styler is not None else None


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
    This decorator works by:

    - catching all inputs that relate to plot style
    - creating a plot, using the passed in plotting function & passing in all non-style arguments
    - passing the style related arguments into a `plot_style` function which applies plot styling

    By default, this function applies styling with the `plot_style` function. Custom
    functions for applying style can be passed in using `plot_style` as a keyword argument.

    The `plot_style` function calls sub-functions for applying different plot elements, including:

    - `axis_styler`: apply style options to an axis
    - `line_styler`: applies style options to lines objects in a plot
    - `collection_styler`: applies style options to collections objects in a plot
    - `custom_style`: applies custom style options

    Each of these sub-functions can be overridden by passing in alternatives.

    To see the full set of style arguments that are supported, run the following code:

    >>> from neurodsp.plts.style import check_style_options
    >>> check_style_options()
    Valid style arguments:
        Axis          title, xlabel, ylabel, xlim, ylim, xticks, yticks, xticklabels, yticklabels, minorticks
        Line          alpha, lw, linewidth, ls, linestyle, marker, ms, markersize
        Collection    alpha, edgecolor
        Custom        title_fontsize, label_size, tick_labelsize, legend_size, legend_loc, tight_layout
    """

    @wraps(func)
    def decorated(*args, **kwargs):

        # Grab a custom style function, if provided, and grab any provided style arguments
        style_func = kwargs.pop('plot_style', plot_style)
        style_args = kwargs.pop('style_args', STYLE_ARGS)
        style_kwargs = {key : kwargs.pop(key) for key in style_args if key in kwargs}

        # Check how many lines are already on the plot, if it exists already
        n_lines_pre = len(kwargs['ax'].lines) if 'ax' in kwargs and kwargs['ax'] is not None else 0

        # Create the plot
        func(*args, **kwargs)

        # Get plot axis, if a specific one was provided, or if not, grab the current axis
        cur_ax = kwargs['ax'] if 'ax' in kwargs and kwargs['ax'] is not None else plt.gca()

        # Check how many lines were added to the plot, and make info available to plot styling
        n_lines_apply = len(cur_ax.lines) - n_lines_pre
        style_kwargs['n_lines_apply'] = n_lines_apply

        # Apply the styling function
        style_func(cur_ax, **style_kwargs)

    return decorated
