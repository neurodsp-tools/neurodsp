"""Decorators for neurodsp.sim."""

from functools import wraps

from neurodsp.sim.utils import demean, normalize_variance

###################################################################################################
###################################################################################################

def normalize(func, **kwargs):

    @wraps(func)
    def decorated(*args, **kwargs):

        # Grab variance & mean as possible kwargs, with default values if not
        variance = kwargs.pop('variance', 1.)
        mean = kwargs.pop('mean', 0.)

        # Call sim function, and unpack to get sig variable, if there are multiple returns
        out = func(*args, **kwargs)
        sig = out[0] if isinstance(out, tuple) else out

        # Apply variance & mean transformations
        if variance is not None:
            sig = normalize_variance(sig, variance=variance)
        if mean is not None:
            sig = demean(sig, mean=mean)

        # Return sig & other outputs, if there were any, or just sig otherwise
        return (sig, out[1:]) if isinstance(out, tuple) else sig

    return decorated
