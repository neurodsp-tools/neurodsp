"""
swm.py
Identify the waveform shape of neural oscillations using the sliding window matching algorithm
"""

import numpy as np


def sliding_window_matching(x, Fs, L, G,
                            max_iterations=500, T=1, window_starts_custom=None):
    """
    Find recurring patterns in a time series using the
    sliding window matching algorithm

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Fs : float
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
    L_samp = int(L * Fs)
    G_samp = int(G * Fs)

    # Initialize window positions, separated by 2*G
    if window_starts_custom is None:
        window_starts = np.arange(0, len(x) - L_samp, 2 * G_samp)
    else:
        window_starts = window_starts_custom
    N_windows = len(window_starts)

    # Calculate initial cost
    J = np.zeros(max_iterations)
    J[0] = _compute_J(x, window_starts, L_samp)

    # Randomly sample windows with replacement
    random_window_idx = np.random.choice(range(N_windows), size=max_iterations)

    # For each iteration, randomly replace a window with a new window
    # to improve cross-window similarity
    iter_num = 1
    while iter_num < max_iterations:
        # Pick a random window position
        window_idx_replace = random_window_idx[iter_num]

        # Find a new allowed position for the window
        window_starts_temp = np.copy(window_starts)
        window_starts_temp[window_idx_replace] = _find_new_windowidx(
            window_starts, G_samp, L_samp, len(x) - L_samp)

        # Calculate the cost
        J_temp = _compute_J(x, window_starts_temp, L_samp)

        # Calculate the change in cost function
        deltaJ = J_temp - J[iter_num - 1]

        # Calculate the acceptance probability
        p_accept = np.exp(-deltaJ / float(T))

        # Accept update to J with a certain probability
        if np.random.rand() < p_accept:
            # Update J
            J[iter_num] = J_temp
            # Update X
            window_starts = window_starts_temp
        else:
            # Update J
            J[iter_num] = J[iter_num - 1]

        # Update iteration number
        iter_num += 1

    # Calculate average window
    avg_window = np.zeros(L_samp)
    for w in range(N_windows):
        avg_window = avg_window + x[window_starts[w]:window_starts[w] + L_samp]
    avg_window = avg_window / float(N_windows)

    return avg_window, window_starts, J


def _compute_J(x, window_starts, L_samp):
    """Compute the cost, which is proportional to the
    difference between pairs of windows"""

    # Get all windows and zscore them
    N_windows = len(window_starts)
    windows = np.zeros((N_windows, L_samp))
    for w in range(N_windows):
        temp = x[window_starts[w]:window_starts[w] + L_samp]
        windows[w] = (temp - np.mean(temp)) / np.std(temp)

    # Calculate distances for all pairs of windows
    d = []
    for i in range(N_windows):
        for j in range(i + 1, N_windows):
            window_diff = windows[i] - windows[j]
            d_temp = np.sum(window_diff**2) / float(L_samp)
            d.append(d_temp)
    # Calculate cost, the average difference, roughly
    J = np.sum(d) / float(2 * (N_windows - 1))
    return J


def _find_new_windowidx(window_starts, G_samp, L_samp, N_samp, tries_limit=1000):
    """Find a new sample for the starting window"""

    found = False
    N_tries = 0
    while found is False:
        # Generate a random sample
        new_samp = np.random.randint(N_samp)
        # Check how close the sample is to other window starts
        dists = np.abs(window_starts - new_samp)
        if np.min(dists) > G_samp:
            return new_samp
        else:
            N_tries += 1
            if N_tries > tries_limit:
                raise RuntimeError('SWM algorithm has difficulty finding a new window. Increase the spacing parameter, G.')
