"""Checker functions."""

###################################################################################################
###################################################################################################

def check_param(param, label, bounds):
    """Check a parameter value.

    Parameters
    ----------
    param : float
        Parameter value to check.
    label : str
        Label of the parameter being checked.
    bounds : list of [float, float]
       Bounding range of valid values for the given parameter.

    Raises
    ------
    ValueError
        If a parameter that is being checked is out of range.
    """

    if (param < bounds[0]) or (param > bounds[1]):
        msg = "The provided value for the {} parameter is out of bounds. ".format(label) + \
        "It should be between {:1.1f} and {:1.1f}.".format(*bounds)
        raise ValueError(msg)
