"""Convert between aperiodic and related measures."""

###################################################################################################
###################################################################################################

def convert_exponent_alpha(exponent):
    """Convert a powerlaw exponent to the expected DFA alpha value.

    Parameters
    ----------
    exponent : float
        Exponent value for a 1/f distribution.

    Returns
    -------
    alpha : float
        Predicted alpha value for the given exponent value.

    References
    ----------
    .. [1] Schaefer, A., Brach, J. S., Perera, S., & Sejdić, E. (2014). A comparative analysis
           of spectral exponent estimation techniques for 1/fβ processes with applications to
           the analysis of stride interval time series. Journal of Neuroscience Methods, 222, 118–130.
           https://doi.org/10.1016/j.jneumeth.2013.10.017
    """

    return (-exponent + 1) / 2


def convert_alpha_exponent(alpha):
    """Convert a DFA alpha value to the expected powerlaw exponent.

    Parameters
    ----------
    alpha : float
        Alpha value from a detrended fluctuation analysis.

    Returns
    -------
    exponent : float
        Predicted exponent value for the given alpha value.

    References
    ----------
    .. [1] Schaefer, A., Brach, J. S., Perera, S., & Sejdić, E. (2014). A comparative analysis
           of spectral exponent estimation techniques for 1/fβ processes with applications to
           the analysis of stride interval time series. Journal of Neuroscience Methods, 222, 118–130.
           https://doi.org/10.1016/j.jneumeth.2013.10.017
    """

    return -2 * alpha + 1


def convert_exponent_hfd(exponent):
    """Convert exponent to expected Higuchi fractal dimension value.

    Parameters
    ----------
    exponent : float
        Exponent value.

    Returns
    -------
    hfd : float
        Predicted Higuchi fractal dimension value.

    Notes
    -----
    This works for exponents between {1, 3} (inclusive).
    As a special case, if exp is 0, D is 2.

    References
    ----------
    .. [1] Cervantes-De la Torre, F., González-Trejo, J. I., Real-Ramírez, C. A., &
           Hoyos-Reyes, L. F. (2013). Fractal dimension algorithms and their application
           to time series associated with natural phenomena. Journal of Physics: Conference
           Series, 475, 012002. https://doi.org/10.1088/1742-6596/475/1/012002
    """

    if exponent == 0:
        hfd = 2
    elif exponent >= 1 and exponent <= 3:
        hfd = (5 - exponent) / 2
    else:
        msg = 'Conversion not supported for given exponent value.'
        raise ValueError(msg)

    return hfd
