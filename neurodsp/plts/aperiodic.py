"""Plotting functions for neurodsp.aperiodic."""

from neurodsp.plts.style import style_plot
from neurodsp.plts.utils import check_ax, savefig, prepare_multi_plot

####################################################################################################
####################################################################################################

@savefig
@style_plot
def plot_autocorr(timepoints, autocorrs, labels=None, colors=None, ax=None, **kwargs):
    """Plot autocorrelation results.

    Parameters
    ----------
    timepoints : 1d array
        Time points, in samples, at which autocorrelations are computed.
    autocorrs : array
        Autocorrelation values, across time lags.
    labels : str or list of str, optional
        Labels for each time series.
    colors : str or list of str
        Colors to use to plot lines.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments for customizing the plot.
    """

    ax = check_ax(ax, figsize=kwargs.pop('figsize', (6, 5)))

    for time, ac, label, color in zip(*prepare_multi_plot(timepoints, autocorrs, labels, colors)):
        ax.plot(time, ac, label=label, color=color)

    ax.set(xlabel='Lag (Samples)', ylabel='Autocorrelation')
