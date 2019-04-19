"""Decorators for neurodsp.sim."""

from functools import wraps

import numpy as np

from neurodsp.utils.norm import demean, normalize_variance

###################################################################################################
###################################################################################################

def normalize(func, **kwargs):
    """Decorator function to normalize the first output of the wrapped function."""

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


def multidim(func, *args, **kwargs):
    """Decorator function to apply the wrapped function across dimensions."""

    @wraps(func)
    def decorated(sig, *args, **kwargs):

        if sig.ndim == 1:
            out = func(sig, *args, **kwargs)

        elif sig.ndim == 2:
            out = np.vstack([func(dat, *args, **kwargs) for dat in sig])

        return out

    return decorated
