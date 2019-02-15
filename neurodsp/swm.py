"""Identify the waveform shape of neural oscillations using the sliding window matching algorithm."""

import numpy as np

###################################################################################################
###################################################################################################

def sliding_window_matching(sig, fs, win_len, win_spacing, max_iterations=500,
                            temperature=1, window_starts_custom=None):
    """Find recurring patterns in a time series using the sliding window matching algorithm.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
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
    window_starts_custom : np.ndarray (1d), optional
        Pre-set locations of initial windows (instead of evenly spaced by 2G).

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
    Gips, B., Bahramisharif, A., Lowet, E., Roberts, M. J., de Weerd, P.,
    Jensen, O., & van der Eerden, J. (2017). Discovering recurring
    patterns in electrophysiological recordings.
    Journal of Neuroscience Methods, 275, 66-79.
    MATLAB code: https://github.com/bartgips/SWM

    Notes
    -----
    * Apply a highpass filter if looking at high frequency activity,
      so that it does not converge on a low frequency motif.
    * win_len and win_spacing should be chosen to be about the size of the motif of
      interest, and the N derived should be about the number of occurrences.
    """

    # Compute window length and spacing in samples
    win_n_samps = int(win_len * fs)
    spacing_n_samps = int(win_spacing * fs)

    # Initialize window positions, separated by 2*G
    if window_starts_custom is None:
        window_starts = np.arange(0, len(sig) - win_n_samps, 2 * spacing_n_samps)
    else:
        window_starts = window_starts_custom
    n_windows = len(window_starts)

    # Calculate initial cost
    costs = np.zeros(max_iterations)
    costs[0] = _compute_cost(sig, window_starts, win_n_samps)

    # Randomly sample windows with replacement
    random_window_idx = np.random.choice(range(n_windows), size=max_iterations)

    # For each iteration, randomly replace a window with a new window
    # to improve cross-window similarity
    iter_num = 1
    while iter_num < max_iterations:
        # Pick a random window position
        window_idx_replace = random_window_idx[iter_num]

        # Find a new allowed position for the window
        window_starts_temp = np.copy(window_starts)
        window_starts_temp[window_idx_replace] = _find_new_windowidx(
            window_starts, spacing_n_samps, len(sig) - win_n_samps)

        # Calculate the cost
        cost_temp = _compute_cost(sig, window_starts_temp, win_n_samps)

        # Calculate the change in cost function
        delta_cost = cost_temp - costs[iter_num - 1]

        # Calculate the acceptance probability
        p_accept = np.exp(-delta_cost / float(temperature))

        # Accept update to J with a certain probability
        if np.random.rand() < p_accept:
            # Update costs
            costs[iter_num] = cost_temp
            # Update windows
            window_starts = window_starts_temp
        else:
            # Update costs
            costs[iter_num] = costs[iter_num - 1]

        # Update iteration number
        iter_num += 1

    # Calculate average window
    avg_window = np.zeros(win_n_samps)
    for w_ind in range(n_windows):
        avg_window = avg_window + sig[window_starts[w_ind]:window_starts[w_ind] + win_n_samps]
    avg_window = avg_window / float(n_windows)

    return avg_window, window_starts, costs


def _compute_cost(sig, window_starts, win_n_samps):
    """Compute the cost, which is proportional to the difference between pairs of windows"""

    # Get all windows and zscore them
    n_windows = len(window_starts)
    windows = np.zeros((n_windows, win_n_samps))
    for w_ind in range(n_windows):
        temp = sig[window_starts[w_ind]:window_starts[w_ind] + win_n_samps]
        windows[w_ind] = (temp - np.mean(temp)) / np.std(temp)

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


def _find_new_windowidx(window_starts, spacing_n_samps, n_samp, tries_limit=1000):
    """Find a new sample for the starting window"""

    found = False
    n_tries = 0

    while found is False:

        # Generate a random sample
        new_samp = np.random.randint(n_samp)

        # Check how close the sample is to other window starts
        dists = np.abs(window_starts - new_samp)

        if np.min(dists) > spacing_n_samps:
            return new_samp
        else:
            n_tries += 1
            if n_tries > tries_limit:
                raise RuntimeError('SWM algorithm has difficulty finding a new window. Increase the spacing parameter, G.')
