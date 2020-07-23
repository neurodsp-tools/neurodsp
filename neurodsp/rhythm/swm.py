"""The sliding window matching algorithm for identifying rhythmic components of a neural signal."""

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
        Window length, in seconds.
    win_spacing : float
        Minimum window spacing, in seconds.
    max_iterations : int
        Maximum number of iterations of potential changes in window placement.
    temperature : float
        Temperature parameter. Controls probability of accepting a new window.
    window_starts_custom : 1d array, optional
        Custom pre-set locations of initial windows.

    Returns
    -------
    avg_window : 1d array
        The average waveform in 'sig' in the frequency 'f_range' triggered on 'trigger'.
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
           Matlab Code: https://github.com/bartgips/SWM

    Notes
    -----
    - Apply a highpass filter if looking at high frequency activity, so that it does
      not converge on a low frequency motif.
    - Parameters `win_len` and `win_spacing` should be chosen to be about the size of the
      motif of interest, and the N derived should be about the number of occurrences.

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

    # Randomly sample windows with replacement
    random_window_idx = np.random.choice(range(n_windows), size=max_iterations)

    # Calculate initial cost
    costs = np.zeros(max_iterations)
    costs[0] = _compute_cost(sig, window_starts, win_n_samps)

    for iter_num in range(1, max_iterations):

        # Pick a random window position to randomly replace with a
        # new window to improve cross-window similarity
        window_idx_replace = random_window_idx[iter_num]

        # Find a new allowed position for the window
        window_starts_temp = np.copy(window_starts)
        window_starts_temp[window_idx_replace] = _find_new_window_idx(
            window_starts, spacing_n_samps, len(sig) - win_n_samps)

        # Calculate the cost & the change in the cost function
        cost_temp = _compute_cost(sig, window_starts_temp, win_n_samps)
        delta_cost = cost_temp - costs[iter_num - 1]

        # Calculate the acceptance probability
        p_accept = np.exp(-delta_cost / float(temperature))

        # Accept update to J with a certain probability
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
    """Compute the cost, which is proportional to the difference between pairs of windows."""

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


def _find_new_window_idx(window_starts, spacing_n_samps, n_samp, tries_limit=1000):
    """Find a new sample for the starting window."""

    for n_try in range(tries_limit):

        # Generate a random sample & check how close it is to other window starts
        new_samp = np.random.randint(n_samp)
        dists = np.abs(window_starts - new_samp)

        if np.min(dists) > spacing_n_samps:
            break

    else:
        raise RuntimeError('SWM algorithm has difficulty finding a new window. \
                            Try increasing the spacing parameter.')

    return new_samp
