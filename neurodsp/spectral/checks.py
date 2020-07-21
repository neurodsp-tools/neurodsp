"""Checker functions for neurodsp.spectral."""

###################################################################################################
###################################################################################################

def check_spg_settings(fs, window, nperseg, noverlap):
    """Check settings used for calculating spectrogram.

    Parameters
    ----------
    fs : float
        Sampling rate, in Hz.
    window : str or tuple or array_like
        Desired window to use. See scipy.signal.get_window for a list of available windows.
        If array_like, the array will be used as the window and its length must be nperseg.
    nperseg : int or None
        Length of each segment, in number of samples.
    noverlap : int or None
        Number of points to overlap between segments.

    Returns
    -------
    nperseg : int
        Length of each segment, in number of samples.
    noverlap : int
        Number of points to overlap between segments.
    """

    # Set the nperseg, if not provided
    if nperseg is None:

        # If the window is a string or tuple, defaults to 1 second of data
        if isinstance(window, (str, tuple)):
            nperseg = int(fs)
        # If the window is an array, defaults to window length
        else:
            nperseg = len(window)
    else:
        nperseg = int(nperseg)

    if noverlap is not None:
        noverlap = int(noverlap)

    return nperseg, noverlap
