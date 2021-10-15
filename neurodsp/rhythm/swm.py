"""The sliding window matching algorithm for identifying recurring patterns in a neural signal."""

import numpy as np

from neurodsp.utils.decorators import multidim

###################################################################################################
###################################################################################################

@multidim()
def sliding_window_matching(sig, fs, win_len, win_spacing, max_iterations=100,
                            window_starts_custom=None, var_thresh=None):
    """Find recurring patterns in a time series using the sliding window matching algorithm.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    win_len : float
        Window length, in seconds. This is L in the original paper.
    win_spacing : float
        Minimum spacing between windows, in seconds. This is G in the original paper.
    max_iterations : int, optional, default: 100
        Maximum number of iterations of potential changes in window placement.
    window_starts_custom : 1d array, optional, default: None
        Custom pre-set locations of initial windows.
    var_thresh: float, optional, default: None
        Removes initial windows with variance below a set threshold. This speeds up
        runtime proportional to the number of low variance windows in the data.

    Returns
    -------
    windows : 2d array
        Putative patterns discovered in the input signal.
    window_starts : 1d array
        Indices at which each window begins for the final set of windows.

    Notes
    -----
    - This algorithm is originally described in [1]_. This re-implementation is a minimal,
      modified version of original ([2]_), which has more available options.
    - The `win_len` parameter should be chosen to be about the size of the motif of interest.
      The larger this window size, the more likely the pattern to reflect slower patterns.
    - The `win_spacing` parameter also determines the number of windows that are used.
    - If looking at high frequency activity, you may want to apply a highpass filter,
      so that the algorithm does not converge on a low frequency motif.
    - This version has the following changes to speed up convergence:

      1. Each iteration is similar to an epoch, randomly moving all windows in
         random order. The original implementation randomly selects windows and
         does not guarantee even resampling.
      2. New window acceptance is determined via increased correlation coefficients
         and reduced variance across windows.
      3. Phase optimization / realignment to escape local minima.

    References
    ----------
    .. [1] Gips, B., Bahramisharif, A., Lowet, E., Roberts, M. J., de Weerd, P., Jensen, O., &
           van der Eerden, J. (2017). Discovering recurring patterns in electrophysiological
           recordings. Journal of Neuroscience Methods, 275, 66-79.
           DOI: https://doi.org/10.1016/j.jneumeth.2016.11.001
    .. [2] Matlab Code implementation: https://github.com/bartgips/SWM

    Examples
    --------
    Search for reoccurring patterns using sliding window matching in a simulated beta signal:

    >>> from neurodsp.sim import sim_combined
    >>> components = {'sim_bursty_oscillation' : {'freq' : 20, 'phase' : 'min'},
    ...               'sim_powerlaw' : {'f_range' : (2, None)}}
    >>> sig = sim_combined(10, fs=500, components=components, component_variances=(1, .05))
    >>> windows, starts = sliding_window_matching(sig, fs=500, win_len=0.05,
    ...                                           win_spacing=0.05, var_thresh=.5)
    """

    # Compute window length and spacing in samples
    win_len = int(win_len * fs)
    win_spacing = int(win_spacing * fs)

    # Initialize window positions
    if window_starts_custom is None:
        window_starts = np.arange(0, len(sig) - win_len, win_spacing).astype(int)
    else:
        window_starts = window_starts_custom

    windows = np.array([sig[start:start + win_len] for start in window_starts])

    # Compute new window bounds
    lower_bounds, upper_bounds = _compute_bounds(window_starts, win_spacing, 0, len(sig) - win_len)

    # Remove low variance windows to speed up runtime
    if var_thresh is not None:

        thresh = np.array([np.var(sig[loc:loc + win_len]) > var_thresh for loc in window_starts])

        windows = windows[thresh]
        window_starts = window_starts[thresh]
        lower_bounds = lower_bounds[thresh]
        upper_bounds = upper_bounds[thresh]

    # Modified SWM procedure
    window_idxs = np.arange(len(windows)).astype(int)

    corrs, variance = _compute_cost(sig, window_starts, win_len)
    mae = np.mean(np.abs(windows - windows.mean(axis=0)))

    for _ in range(max_iterations):

        # Randomly shuffle order of windows
        np.random.shuffle(window_idxs)

        for win_idx in window_idxs:

            # Find a new, random window start
            _window_starts = window_starts.copy()
            _window_starts[win_idx] = np.random.choice(np.arange(lower_bounds[win_idx],
                                                                 upper_bounds[win_idx] + 1))

            # Accept new window if correlation increases and variance decreases
            _corrs, _variance = _compute_cost(sig, _window_starts, win_len)

            if _corrs[win_idx].sum() > corrs[win_idx].sum() and  _variance < variance:

                corrs = _corrs.copy()
                variance = _variance
                window_starts = _window_starts.copy()
                lower_bounds, upper_bounds = _compute_bounds(\
                    window_starts, win_spacing, 0, len(sig) - win_len)

        # Phase optimization
        _window_starts = window_starts.copy()

        for shift in np.arange(-int(win_len/2), int(win_len/2)):

            _starts = _window_starts + shift

            # Skip windows shifts that are out-of-bounds
            if (_starts[0] < 0) or (_starts[-1] > len(sig) - win_len):
                continue

            _windows = np.array([sig[start:start + win_len] for start in _starts])

            _mae = np.mean(np.abs(_windows - _windows.mean(axis=0)))

            if _mae < mae:
                window_starts = _starts.copy()
                windows = _windows.copy()
                mae = _mae

        lower_bounds, upper_bounds = _compute_bounds(\
            window_starts, win_spacing, 0, len(sig) - win_len)

    return windows, window_starts


def _compute_cost(sig, window_starts, win_n_samps):
    """Compute the cost, as correlation coefficients and variance across windows.

    Parameters
    ----------
    sig : 1d array
        Time series.
    window_starts : list of int
        The list of window start definitions.
    win_n_samps : int
        The length of each window, in samples.

    Returns
    -------
    corrs : 2d array
        Window correlation matrix.
    variance : float
        Sum of the variance across windows.
    """

    windows = np.array([sig[start:start + win_n_samps] for start in window_starts])

    corrs = np.corrcoef(windows)

    variance = windows.var(axis=1).sum()

    return corrs, variance


def _compute_bounds(window_starts, win_spacing, start, end):
    """Compute bounds on a new window.

    Parameters
    ----------
    window_starts : list of int
        The list of window start definitions.
    win_spacing : float
        Minimum spacing between windows, in seconds.
    start, end : int
        Start and end edges for the possible window.

    Returns
    -------
    lower_bounds, upper_bounds : 1d array
        Computed upper and lower bounds for the window position.
    """

    lower_bounds = window_starts[:-1] + win_spacing
    lower_bounds = np.insert(lower_bounds, 0, start)

    upper_bounds = window_starts[1:] - win_spacing
    upper_bounds = np.insert(upper_bounds, len(upper_bounds), end)

    return lower_bounds, upper_bounds
