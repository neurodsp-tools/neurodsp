"""Checker functions for neurodsp.spectral."""

###################################################################################################
###################################################################################################

def check_spg_settings(fs, window, nperseg, noverlap):
    """Check settings used for calculating spectrogram."""

    # Set the nperseg, if not provided:
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
