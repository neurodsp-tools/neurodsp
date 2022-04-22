"""Convert between aperiodic and related measures."""

from neurodsp.utils.checks import check_param_range, check_param_options

###################################################################################################
###################################################################################################

def convert_exponent_alpha(exponent):
    """Convert a powerlaw exponent to the expected DFA alpha value.

    Parameters
    ----------
    exponent : float
        Aperiodic exponent value, representing a 1/f distribution.

    Returns
    -------
    alpha : float
        Predicted DFA alpha value for the given exponent value.

    References
    ----------
    .. [1] Schaefer, A., Brach, J. S., Perera, S., & Sejdić, E. (2014). A comparative analysis
           of spectral exponent estimation techniques for 1/fβ processes with applications to
           the analysis of stride interval time series. Journal of Neuroscience Methods, 222, 118–130.
           https://doi.org/10.1016/j.jneumeth.2013.10.017
    """

    return (-exponent + 1) / 2.


def convert_alpha_exponent(alpha):
    """Convert a DFA alpha value to the expected powerlaw exponent.

    Parameters
    ----------
    alpha : float
        Alpha value from a detrended fluctuation analysis.

    Returns
    -------
    exponent : float
        Predicted aperiodic exponent value, representing a 1/f distribution.

    References
    ----------
    .. [1] Schaefer, A., Brach, J. S., Perera, S., & Sejdić, E. (2014). A comparative analysis
           of spectral exponent estimation techniques for 1/fβ processes with applications to
           the analysis of stride interval time series. Journal of Neuroscience Methods, 222, 118–130.
           https://doi.org/10.1016/j.jneumeth.2013.10.017
    """

    return -2. * alpha + 1


def convert_exponent_hurst(exponent, fractional_class):
    """Convert a powerlaw exponent to the expected Hurst exponent value.

    Parameters
    ----------
    exponent : float
        Aperiodic exponent value, representing a 1/f distribution.
    fractional_class : {'gaussian', 'brownian'}
        The class of input data that the given exponent value relates to.
        This can be either 'fractional Gaussian noise' or 'fractional Brownian motion.'
        This is required as the conversion differs between the two classes.

    Returns
    -------
    hurst : float
        Predicted Hurst exponent for the given exponent value.

    References
    ----------
    .. [1] Schaefer, A., Brach, J. S., Perera, S., & Sejdić, E. (2014). A comparative analysis
           of spectral exponent estimation techniques for 1/fβ processes with applications to
           the analysis of stride interval time series. Journal of Neuroscience Methods, 222, 118–130.
           https://doi.org/10.1016/j.jneumeth.2013.10.017
    """

    check_param_options(fractional_class, 'fractional_class', ['gaussian', 'brownian'])

    # Switch sign of exponent (assumed to be negative) for this conversion
    exponent = -exponent

    if fractional_class == 'gaussian':
        hurst = (exponent + 1) / 2.
    elif fractional_class == 'brownian':
        hurst = (exponent - 1) / 2.

    return hurst

def convert_hurst_exponent(hurst, fractional_class):
    """Convert a Hurst exponent value to the expected powerlaw exponent.

    Parameters
    ----------
    hurst : float
        Hurst exponent value.
    fractional_class : {'gaussian', 'brownian'}
        The class of input data that the given exponent value relates to.
        This can be either 'fractional Gaussian noise' or 'fractional Brownian motion.'
        This is required as the conversion differs between the two classes.

    Returns
    -------
    exponent : float
        Predicted aperiodic exponent value, representing a 1/f distribution.

    References
    ----------
    .. [1] Schaefer, A., Brach, J. S., Perera, S., & Sejdić, E. (2014). A comparative analysis
           of spectral exponent estimation techniques for 1/fβ processes with applications to
           the analysis of stride interval time series. Journal of Neuroscience Methods, 222, 118–130.
           https://doi.org/10.1016/j.jneumeth.2013.10.017
    """

    check_param_options(fractional_class, 'fractional_class', ['gaussian', 'brownian'])

    if fractional_class == 'gaussian':
        exponent = 2. * hurst - 1
    elif fractional_class == 'brownian':
        exponent = 2. * hurst + 1

    # Convert predicted exponent value to have a negative sign
    exponent = -exponent

    return exponent


def convert_exponent_hfd(exponent):
    """Convert exponent to expected Higuchi fractal dimension value.

    Parameters
    ----------
    exponent : float
        Aperiodic exponent value, representing a 1/f distribution.

    Returns
    -------
    hfd : float
        Predicted Higuchi fractal dimension value.

    Notes
    -----
    This works for exponents between [-1, -3] (inclusive).
    As a special case, if exp is 0, FD is 2.

    References
    ----------
    .. [1] Cervantes-De la Torre, F., González-Trejo, J. I., Real-Ramírez, C. A., &
           Hoyos-Reyes, L. F. (2013). Fractal dimension algorithms and their application
           to time series associated with natural phenomena. Journal of Physics: Conference
           Series, 475, 012002. https://doi.org/10.1088/1742-6596/475/1/012002
    """

    if exponent == 0:
        hfd = 2.
    else:
        # Switch sign of exponent (assumed to be negative) for this conversion
        exponent = -exponent
        check_param_range(exponent, 'exponent', [1, 3])
        hfd = (5 - exponent) / 2.

    return hfd


def convert_hfd_exponent(hfd):
    """Convert Higuchi fractal dimension value to expected 1/f exponent value.

    Parameters
    ----------
    hfd : float
        Higuchi fractal dimension value.

    Returns
    -------
    exponent : float
        Predicted aperiodic exponent value, representing a 1/f distribution.

    Notes
    -----
    This works for Fractal Dimensions between [1, 2] (inclusive).

    References
    ----------
    .. [1] Cervantes-De la Torre, F., González-Trejo, J. I., Real-Ramírez, C. A., &
           Hoyos-Reyes, L. F. (2013). Fractal dimension algorithms and their application
           to time series associated with natural phenomena. Journal of Physics: Conference
           Series, 475, 012002. https://doi.org/10.1088/1742-6596/475/1/012002
    """

    check_param_range(hfd, 'hfd', [1, 2])
    exponent = -2. * hfd + 5
    # Convert predicted exponent value to have a negative sign
    exponent = -exponent

    return exponent
