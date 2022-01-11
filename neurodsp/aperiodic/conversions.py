"""Convert between measures."""

###################################################################################################
###################################################################################################

def convert_exponent_alpha(exp):
    """Convert a powerlaw exponent to the expected DFA alpha value.

    Parameters
    ----------
    exp : float
        Exponent value for a 1/f distribution.

    Returns
    -------
    float
        Predicted alpha value for the given exponent value.
    """

    return (-exp + 1) / 2

def convert_alpha_exponent(alpha):
    """Convert a DFA alpha value to the expected powerlaw exponent.

    Parameters
    ----------
    alpha : float
        Alpha value from a detrended fluctuation analysis.

    Returns
    -------
    float
        Predicted exponent value for the given alpha value.
    """

    return -2 * alpha + 1
