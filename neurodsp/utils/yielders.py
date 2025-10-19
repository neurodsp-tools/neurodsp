"""Utilities, using yield, for iterating across / around signals."""

###################################################################################################
###################################################################################################

def step_over_time(sig, start=0, size=100, step=1):
    """Helper function to yield segments of a signal.

    Parameters
    ----------
    sig : 1d array
        Time series to iterate across.
    start : int, optional, default: 0
        Staring index.
    size : int, optional, default: 100
        Size of each segment to yield.
    step : int, optional, default: 1
        Step size of each iteration.

    Yields
    ------
    segment : 1d array
        Extracted segment of the time series.
    """

    for st in range(start, len(sig)-size, step):
        yield sig[st:st+size]


def step_over_signals(signals):
    """"Step across signals within an array.

    Parameters
    ----------
    signals : 2d array
        Array of signals to iterate across.

    Yields
    ------
    sig : 1d array
        Extracted signal.
    """

    for sig in signals:
        yield sig
