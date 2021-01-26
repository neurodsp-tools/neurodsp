"""The sliding window matching algorithm for identifying recurring patterns in a neural signal."""

import numpy as np

from neurodsp.utils.decorators import multidim

###################################################################################################
###################################################################################################

@multidim()
def sliding_window_matching(sig, fs, win_len, win_spacing, max_iterations=500,
                            temperature=1, window_starts_custom=None):
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
    max_iterations : int, optional, default: 500
        Maximum number of iterations of potential changes in window placement.
    temperature : float, optional, default: 1
        Controls the probability of accepting a new window.
    window_starts_custom : 1d array, optional
        Custom pre-set locations of initial windows.

    Returns
    -------
    avg_window : 1d array
        The average pattern discovered in the input signal.
    window_starts : 1d array
        Indices at which each window begins for the final set of windows.
    costs : 1d array
        Cost function value at each iteration.

    References
    ----------
    .. [1] Gips, B., Bahramisharif, A., Lowet, E., Roberts, M. J., de Weerd, P., Jensen, O., &
           van der Eerden, J. (2017). Discovering recurring patterns in electrophysiological
           recordings. Journal of Neuroscience Methods, 275, 66-79.
           DOI: 10.1016/j.jneumeth.2016.11.001
    .. [2] Matlab Code implementation: https://github.com/bartgips/SWM

    Notes
    -----
    - The `win_len` parameter should be chosen to be about the size of the motif of interest.
      The larger this window size, the more likely the pattern to reflect slower patterns.
    - The `win_spacing` parameter also determines the number of windows that are used.
    - If looking at high frequency activity, you may want to apply a highpass filter,
      so that the algorithm does not converge on a low frequency motif.
    - This implementation is a minimal version, as compared to the original implementation
      in [2], which has more available options.
    - This version may also be much smaller than the original, as this implementation does not
      currently include the Markov Chain Monte Carlo sampling to calculate distances.

    Examples
    --------
    Search for reoccuring patterns using sliding window matching in a simulated beta signal:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {'f_range': (2, None)},
    ...                                'sim_bursty_oscillation': {'freq': 20,
    ...                                                           'enter_burst': .25,
    ...                                                           'leave_burst': .25}})
    >>> avg_window, window_starts, costs = sliding_window_matching(sig, fs=500, win_len=0.05,
    ...                                                            win_spacing=0.20)
    """

    # Compute window length and spacing in samples
    win_n_samps = int(win_len * fs)
    spacing_n_samps = int(win_spacing * fs)

    # Initialize window positions
    if window_starts_custom is None:
        window_starts = np.arange(0, len(sig) - win_n_samps, 2 * spacing_n_samps)
    else:
        window_starts = window_starts_custom
    n_windows = len(window_starts)


    # Calculate initial cost
    costs = np.zeros(max_iterations)
    costs[0] = _compute_cost(sig, window_starts, win_n_samps)

    for iter_num in range(1, max_iterations):

        # Find a new, random window start
        window_starts_temp = _find_new_window_idx(np.copy(window_starts), len(sig) - win_n_samps,
                                                  spacing_n_samps)

        # Calculate the cost & the change in the cost function
        cost_temp = _compute_cost(sig, window_starts_temp, win_n_samps)
        delta_cost = cost_temp - costs[iter_num - 1]

        # Calculate the acceptance probability
        p_accept = np.exp(-delta_cost / float(temperature))

        # Accept update, based on the calculated acceptance probability
        if np.random.rand() < p_accept:

            # Update costs & windows
            costs[iter_num] = cost_temp
            window_starts = window_starts_temp

        else:

            # Update costs
            costs[iter_num] = costs[iter_num - 1]

    # Calculate average window
    avg_window = np.zeros(win_n_samps)
    for w_ind in range(n_windows):
        avg_window = avg_window + sig[window_starts[w_ind]:window_starts[w_ind] + win_n_samps]
    avg_window = avg_window / float(n_windows)

    return avg_window, window_starts, costs


def _compute_cost(sig, window_starts, win_n_samps):
    """Compute the cost, which is proportional to the difference between pairs of windows.

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
    cost: float
        The calculated cost of the current set of windows.
    """

    # Get all windows and z-score them
    n_windows = len(window_starts)
    windows = np.zeros((n_windows, win_n_samps))
    for ind, window in enumerate(window_starts):
        temp = sig[window:window_starts[ind] + win_n_samps]
        windows[ind] = (temp - np.mean(temp)) / np.std(temp)

    # Calculate distances for all pairs of windows
    dists = []
    for ind1 in range(n_windows):
        for ind2 in range(ind1 + 1, n_windows):
            window_diff = windows[ind1] - windows[ind2]
            dist_temp = np.sum(window_diff**2) / float(win_n_samps)
            dists.append(dist_temp)

    # Calculate cost function, which is the average difference, roughly
    cost = np.sum(dists) / float(2 * (n_windows - 1))

    return cost


def _find_new_window_idx(windows, max_start, spacing_n_samps, tries_limit=1000):
    """Find a new sample for the starting window.

    Parameters
    ----------
    windows : 1d array
        Indices at which each window begins.
    max_start : int
        The largest possible start index for the window.
    spacing_n_samps : int
        The minimum spacing required between windows, in samples.
    tries_limit : int, optional, default: False
        The maximum number of iterations to attempt to find a new window.

    Returns
    -------
    windows : 1d array
        Indices, including the updated window, at which each window begins.
    """

    new_win = None

    for _ in range(tries_limit):

        # Randomly select current window
        current_idx = np.random.choice(range(len(windows)))
        current_win = windows[current_idx]

        # Find adjacent window starts
        if current_idx != 0:
            lower_bound = windows[current_idx - 1] + spacing_n_samps
        else:
            lower_bound = 0

        if current_idx != len(windows) - 1:
            upper_bound = windows[current_idx + 1] - spacing_n_samps
        else:
            upper_bound = max_start

        # Create allowed random positions/indices
        new_wins = np.arange(lower_bound, upper_bound + 1)
        new_wins = new_wins[np.where((new_wins >= 0) & (new_wins <= max_start))[0]]

        # Drop the current window index so it can't be randomly sampled
        new_wins = np.delete(new_wins, np.where(new_wins == current_win)[0][0])

        if len(new_wins) == 0:
            # All new samples are too close to adjacent window starts, try another random position
            continue

        # Randomly select allowed sample position
        new_win= np.random.choice(new_wins)

        break

    if new_win is None:
        raise RuntimeError('SWM algorithm has difficulty finding a new window. \
                            Try increasing the spacing parameter.')

    windows[current_idx] = new_win

    return windows
