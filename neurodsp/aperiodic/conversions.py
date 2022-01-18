"""Convert between aperiodic and related measures."""

###################################################################################################
###################################################################################################

def convert_exponent_alpha(exponent):
    """Convert a powerlaw exponent to the expected DFA alpha value.

    Parameters
    ----------
    exp : float
        Exponent value for a 1/f distribution.

    Returns
    -------
    alpha : float
        Predicted alpha value for the given exponent value.
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
    From F Cervantes-De la Torre et al, 2013
    """

    if exponent == 0:
        hfd = 2
    elif exponent >= 1 and exponent <= 3:
        hfd = (5 - exponent) / 2
    else:
        msg = 'Conversion not supported for given exponent value.'
        raise ValueError(msg)

    return hfd
