"""Utility functions for simulations."""

###################################################################################################
###################################################################################################

def check_osc_def(n_seconds, fs, freq):
    """Check whether a requested oscillation definition will have an integer number of cycles.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    freq : float
        Oscillation frequency.

    Returns
    -------
    bool
        Whether the definition will have an integer number of cycles.
    """

    # Sampling rate check: check if the number of samples per cycle is an integer
    srate_check = (fs/freq).is_integer()

    # Time check: check if signal length matches an integer number of cycles
    time_check = (n_seconds * freq).is_integer()

    return srate_check and time_check
