"""Fluctuation analyses to measure fractal properties of time series."""

import numpy as np
import scipy as sp

###################################################################################################
###################################################################################################

def compute_fluctuations(sig, fs, n_scales=10, min_scale=0.01, max_scale=1.0, deg=1, method='dfa'):
    """Compute a fluctuation analysis on a signal.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float, Hz
        Sampling rate, in Hz.
    n_scales : int, optional, default=10
        Number of scales to estimate fluctuations over.
    min_scale : float, optional, default=0.01
        Shortest scale to compute over, in seconds.
    max_scale : float, optional, default=1.0
        Longest scale to compute over, in seconds.
    deg : int, optional, default=1
        Polynomial degree for detrending. Only used for DFA.

        - 1 for regular DFA
        - 2 or higher for generalized DFA
    method : {'dfa', 'rs'}
        Method to use to compute fluctuations:

        - 'dfa' : detrended fluctuation
        - 'rs' : rescaled range

    Returns
    -------
    t_scales : 1d array
        Time-scales over which fluctuation measures were computed.
    fluctuations : 1d array
        Average fluctuation at each scale.
    exp : float
        Slope of line in loglog when plotting t_scales against fluctuations.
        This is the alpha value for DFA, or the Hurst exponent for rescaled range.

    Notes
    -----
    These analysis compute fractal properties by analyzing fluctuations across windows.

    Overall, these approaches involve dividing the time-series into non-overlapping
    windows at log-spaced scales and computing a fluctuation measure across windows.
    The relationship of this fluctuation measure across window sizes provides
    information on the fractal properties of a signal.

    Measures available are:
    - DFA: detrended fluctuation analysis
        - computes ordinary least squares fits across signal windows
    - RS: rescaled range
        - computes the range of signal windows, divided by the standard deviation

    Empirically, DFA seems to work for a larger range of exponent values.
    """

    # Get log10 equi-spaced scales and translate that into window lengths
    t_scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)
    win_lens = np.round(t_scales * fs).astype('int')

    # Calculate culmulative sum of the signal
    sig_walk = sp.cumsum(sig - np.mean(sig))

    # Step through each scale and get RMS
    fluctuations = np.zeros_like(t_scales)
    for idx, win_len in enumerate(win_lens):
        if method is 'dfa':
            fluctuations[idx] = compute_detrended_fluctuation(sig_walk, win_len=win_len, deg=deg)
        elif method is 'rs':
            fluctuations[idx] = compute_rescaled_range(sig_walk, win_len=win_len)
        else:
            raise ValueError('Fluctuation method not understood.')

    # Calculate the relationship between between fluctuations & time scales
    exp = np.polyfit(np.log10(t_scales), np.log10(fluctuations), deg=1)[0]

    return t_scales, fluctuations, exp


def compute_rescaled_range(sig, win_len):
    """Compute rescaled range of a given time-series at a given scale.

    Parameters
    ----------
    sig : 1d array
        Time series.
    win_len : int
        Window length for each rescaled range computation.

    Returns
    -------
    rs : float
        Average rescaled range over windows.
    """

    # Gather windows & vectorize, so we can call math functions in one go
    n_win = int(np.floor(len(sig) / win_len))
    sig_rect = np.reshape(sig[:n_win * win_len], (n_win, win_len)).T

    # get back the sig by taking the derivative of sig_walk
    X = np.concatenate((sig[:1], np.diff(sig)))
    X_rect = np.reshape(X[:n_win * win_len], (n_win, win_len)).T

    # Calculate rescaled range as range divided by std, and take mean across windows
    rs_win = np.ptp(sig_rect, axis=0) / np.std(X_rect, axis=0)
    rs = np.mean(rs_win)

    return rs


def compute_detrended_fluctuation(sig, win_len, deg=1):
    """Compute detrended fluctuation of a time-series at the given window length.

    Parameters
    ----------
    sig : 1d array
        Time series.
    win_len : int
        Window length for each detrended fluctuation fit.
    deg : int, optional, default=1
        Polynomial degree for detrending.

    Returns
    -------
    det_fluc : float
        Measured detrended fluctuaiton, as the average error fits of the window.
    """

    # Gather windows & vectorize, so we can call math functions in one go
    n_win = int(np.floor(len(sig) / win_len))
    sig_rect = np.reshape(sig[:n_win * win_len], (n_win, win_len)).T

    # Calculate local trend, as the line of best fit within the time window
    _, fluc, _, _, _ = np.polyfit(np.arange(win_len), sig_rect, deg=deg, full=True)

    # Convert to root-mean squared error, from squared error
    det_fluc = np.mean((fluc / win_len)**0.5)

    return det_fluc
