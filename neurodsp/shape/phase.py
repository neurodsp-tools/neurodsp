"""
phase.py
Estimate the phase of an oscillation using a waveform-based approach
"""

import numpy as np


def extrema_interpolated_phase(x, Ps, Ts, zeroxR=None, zeroxD=None):
    """
    Use peaks (phase 0) and troughs (phase pi/-pi) to estimate
    instantaneous phase. Also use rise and decay zerocrossings
    (phase -pi/2 and pi/2, respectively) if specified.

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ps : array-like 1d
        time points of oscillatory peaks
    Ts : array-like 1d
        time points of oscillatory troughs
    zeroxR : array-like 1d
        indices at which oscillatory rising zerocrossings occur
    zeroxD : array-like 1d
        indices at which oscillatory decaying zerocrossings occur

    Returns
    -------
    pha : array-like 1d
        instantaneous phase

    Notes
    -----
    Sometimes, due to noise, extrema and zerocrossing estimation
    is poor, and for example, the same index may be assigned to
    both a peak and a decaying zerocrossing. Because of this,
    we first assign phase values by zerocrossings, and then
    may overwrite them with extrema phases.
    """

    # Initialize phase arrays
    # 2 phase arrays: trough pi and trough -pi
    L = len(x)
    t = np.arange(L)
    pha_tpi = np.zeros(L) * np.nan
    pha_tnpi = np.zeros(L) * np.nan

    # If specified, assign phases to zerocrossings
    if zeroxR is not None:
        pha_tpi[zeroxR] = -np.pi / 2
        pha_tnpi[zeroxR] = -np.pi / 2
    if zeroxD is not None:
        pha_tpi[zeroxD] = np.pi / 2
        pha_tnpi[zeroxD] = np.pi / 2

    # Define phases
    pha_tpi[Ps] = 0
    pha_tpi[Ts] = np.pi
    pha_tnpi[Ps] = 0
    pha_tnpi[Ts] = -np.pi

    # Interpolate to find all phases
    pha_tpi = np.interp(t, t[~np.isnan(pha_tpi)], pha_tpi[~np.isnan(pha_tpi)])
    pha_tnpi = np.interp(t, t[~np.isnan(pha_tnpi)], pha_tnpi[~np.isnan(pha_tnpi)])

    # For the phase time series in which the trough is negative pi:
    # Replace the decaying periods with these periods in the phase time
    # series in which the trough is pi
    diffs = np.diff(pha_tnpi)
    diffs = np.append(diffs, 99)
    pha_tnpi[diffs < 0] = pha_tpi[diffs < 0]

    # Assign the periods before the first empirical phase timepoint to NaN
    diffs = np.diff(pha_tnpi)
    first_empirical_idx = next(i for i, xi in enumerate(diffs) if xi > 0)
    pha_tnpi[:first_empirical_idx] = np.nan

    # Assign the periods after the last empirical phase timepoint to NaN
    diffs = np.diff(pha_tnpi)
    last_empirical_idx = next(i for i, xi in enumerate(diffs[::-1]) if xi > 0)
    pha_tnpi[-last_empirical_idx + 1:] = np.nan

    return pha_tnpi
