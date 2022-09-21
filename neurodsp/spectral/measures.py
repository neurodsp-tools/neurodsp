""""Compute spectral power related measures."""

from neurodsp.utils.core import get_avg_func
from neurodsp.spectral.utils import trim_spectrum

###################################################################################################
###################################################################################################

def compute_absolute_power(freqs, powers, band, method='sum'):
    """Compute absolute power for a given frequency band.

    Parameters
    ----------
    freqs : 1d array
        Frequency values.
    powers : 1d array
        Power spectrum power values.
    band : list of [float, float]
        Band definition.
    method : {'sum', 'mean', 'median'}, optional
        Method to use to compute power across the band.

    Returns
    -------
    abs_power : float
        Computed absolute power.

    Examples
    --------
    Compute absolute alpha power in a simulated signal:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.spectral import compute_spectrum
    >>> sig = sim_combined(10, 500, {'sim_powerlaw': {}, 'sim_oscillation': {'freq': 10}})
    >>> freqs, powers = compute_spectrum(sig, fs=500)
    >>> abs_power = compute_absolute_power(freqs, powers, [8, 12])
    """

    _, band_powers = trim_spectrum(freqs, powers, band)
    abs_power = get_avg_func(method)(band_powers)

    return abs_power


def compute_relative_power(freqs, powers, band, method='sum', norm_range=None):
    """Compute relative power for a given frequency band.

    Parameters
    ----------
    freqs : 1d array
        Frequency values.
    powers : 1d array
        Power spectrum power values.
    band : list of [float, float]
        Band definition.
    method : {'sum', 'mean', 'median'}, optional
        Method to use to compute power across the band.
    norm_range : list of [float, float], optional
        Frequency range to use to compute total power.
        If not provided, the whole spectrum is used.

    Returns
    -------
    rel_power : float
        Computed relative power.

    Examples
    --------
    Compute relative alpha power in a simulated signal:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.spectral import compute_spectrum
    >>> sig = sim_combined(10, 500, {'sim_powerlaw': {}, 'sim_oscillation': {'freq': 10}})
    >>> freqs, powers = compute_spectrum(sig, fs=500)
    >>> abs_power = compute_relative_power(freqs, powers, [8, 12])
    """

    band_power = compute_absolute_power(freqs, powers, band, method)

    total_band = [freqs.min(), freqs.max()] if not norm_range else norm_range
    total_power = compute_absolute_power(freqs, powers, total_band, method)

    rel_power = band_power / total_power

    return rel_power


def compute_band_ratio(freqs, powers, low_band, high_band, method='mean'):
    """Calculate band ratio measure between two predefined frequency ranges.

    Parameters
    ----------
    freqs : 1d array
        Frequency values.
    powers : 1d array
        Power spectrum power values.
    low_band : list of [float, float]
        Band definition for the lower band.
    high_band : list of [float, float]
        Band definition for the upper band.
    method : {'mean', 'median', 'sum'}, optional
        Method to use to compute power across the band.

    Outputs
    -------
    ratio : float
        Band ratio.

    Notes
    -----
    This function is provided as a convenience function, for computing band ratio measures
    in order to, for example, compare to other measures. In general, band ratio measures
    are not recommended, as they conflate aperiodic and periodic features (see [1] for more).

    References
    ----------
    .. [1] Donoghue, T., Dominguez, J., & Voytek, B. (2020). Electrophysiological Frequency
           Band Ratio Measures Conflate Periodic and Aperiodic Neural Activity. ENeuro, 7(6),
           ENEURO.0192-20.2020. DOI: https://doi.org/10.1523/ENEURO.0192-20.2020

    Examples
    --------
    Compute theta/beta band ratio in a simulated signal:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.spectral import compute_spectrum
    >>> sig = sim_combined(10, 500, {'sim_powerlaw': {}, 'sim_oscillation': {'freq': 10}})
    >>> freqs, powers = compute_spectrum(sig, fs=500)
    >>> abs_power = compute_band_ratio(freqs, powers, [4, 8], [13, 30])
    """

    # Compute the power in each band
    low_band_power = compute_absolute_power(freqs, powers, low_band, method)
    high_band_power = compute_absolute_power(freqs, powers, high_band, method)

    # Calculate the ratio between bands
    ratio = low_band_power / high_band_power

    return ratio
