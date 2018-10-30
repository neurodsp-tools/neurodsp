"""Identify the waveform shape of neural oscillations using the sliding window matching algorithm."""

import numpy as np

###################################################################################################
###################################################################################################

def sliding_window_matching(x, s_rate, L, G, max_iterations=500, T=1, window_starts_custom=None):
    """Find recurring patterns in a time series using the sliding window matching algorithm.

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    s_rate : float
        sampling rate (samples per second)
    L : float
        window length (seconds)
    G : float
        minimum window spacing (seconds)
    T : float
        temperature parameter. Controls probability of accepting a new window
    max_iterations : int
        Maximum number of iterations of potential changes in window placement
    window_starts_custom : np.ndarray (1d)
        Pre-set locations of initial windows (instead of evenly spaced by 2G)

    Returns
    -------
    avg_window : np.ndarray (1d)
        the average waveform in 'x' in the frequency 'f_range' triggered on 'trigger'
    window_starts : np.ndarray (1d)
        indices at which each window begins for the final set of windows
    J : np.ndarray (1d)
        Cost function value at each iteration

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
      so that it does not converge on a low frequency motif
    * L and G should be chosen to be about the size of the motif of interest,
       and the N derived should be about the number of occurrences
    """

    # Compute window length and spacing in samples
    l_samp = int(L * s_rate)
    g_samp = int(G * s_rate)

    # Initialize window positions, separated by 2*G
    if window_starts_custom is None:
        window_starts = np.arange(0, len(x) - l_samp, 2 * g_samp)
    else:
        window_starts = window_starts_custom
    n_windows = len(window_starts)

    # Calculate initial cost
    J = np.zeros(max_iterations)
    J[0] = _compute_J(x, window_starts, l_samp)

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
            window_starts, g_samp, l_samp, len(x) - l_samp)

        # Calculate the cost
        j_temp = _compute_J(x, window_starts_temp, l_samp)

        # Calculate the change in cost function
        delta_j = j_temp - J[iter_num - 1]

        # Calculate the acceptance probability
        p_accept = np.exp(-delta_j / float(T))

        # Accept update to J with a certain probability
        if np.random.rand() < p_accept:
            # Update J
            J[iter_num] = j_temp
            # Update X
            window_starts = window_starts_temp
        else:
            # Update J
            J[iter_num] = J[iter_num - 1]

        # Update iteration number
        iter_num += 1

    # Calculate average window
    avg_window = np.zeros(l_samp)
    for w in range(n_windows):
        avg_window = avg_window + x[window_starts[w]:window_starts[w] + l_samp]
    avg_window = avg_window / float(n_windows)

    return avg_window, window_starts, J


def _compute_J(x, window_starts, l_samp):
    """Compute the cost, which is proportional to the difference between pairs of windows"""

    # Get all windows and zscore them
    n_windows = len(window_starts)
    windows = np.zeros((n_windows, l_samp))
    for w in range(n_windows):
        temp = x[window_starts[w]:window_starts[w] + l_samp]
        windows[w] = (temp - np.mean(temp)) / np.std(temp)

    # Calculate distances for all pairs of windows
    d = []
    for i in range(n_windows):
        for j in range(i + 1, n_windows):
            window_diff = windows[i] - windows[j]
            d_temp = np.sum(window_diff**2) / float(l_samp)
            d.append(d_temp)
    # Calculate cost, the average difference, roughly
    J = np.sum(d) / float(2 * (n_windows - 1))
    return J


def _find_new_windowidx(window_starts, g_samp, l_samp, N_samp, tries_limit=1000):
    """Find a new sample for the starting window"""

    found = False
    n_tries = 0
    while found is False:
        # Generate a random sample
        new_samp = np.random.randint(N_samp)
        # Check how close the sample is to other window starts
        dists = np.abs(window_starts - new_samp)
        if np.min(dists) > g_samp:
            return new_samp
        else:
            n_tries += 1
            if n_tries > tries_limit:
                raise RuntimeError('SWM algorithm has difficulty finding a new window. Increase the spacing parameter, G.')
