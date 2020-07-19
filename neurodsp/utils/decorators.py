"""Decorators."""

from functools import wraps

import numpy as np

from neurodsp.utils.norm import normalize_sig

###################################################################################################
###################################################################################################

def normalize(func, **kwargs):
    """Decorator function to normalize the first output of the wrapped function."""

    @wraps(func)
    def decorated(*args, **kwargs):

        # Grab variance & mean as possible kwargs, with default values if not
        mean = kwargs.pop('mean', 0.)
        variance = kwargs.pop('variance', 1.)

        # Call sim function, and unpack to get sig variable, if there are multiple returns
        out = func(*args, **kwargs)
        sig = out[0] if isinstance(out, tuple) else out

        # Normalize signal, applying mean and variance transformations
        sig = normalize_sig(sig, mean, variance)

        # Return sig & other outputs, if there were any, or just sig otherwise
        return (sig, out[1:]) if isinstance(out, tuple) else sig

    return decorated


def multidim(select=[]):
    """Decorator function to apply the wrapped function across dimensions.

    Parameters
    ----------
    select : list of int, optional
        List of indices of outputs to sub-select a single instance from.
    """

    def decorator(func, *args, **kwargs):

        @wraps(func)
        def wrapper(sig, *args, **kwargs):

            if sig.ndim == 1:
                out = func(sig, *args, **kwargs)

            elif sig.ndim == 2:

                # Apply func across rows of the input data
                outs = [func(dat, *args, **kwargs) for dat in sig]

                if isinstance(outs[0], tuple):

                    # Collect together associated outputs from each,
                    #   in case there are multiple outputs
                    out = [np.stack([dat[n_out] for dat in outs]) \
                        for n_out in range(len(outs[0]))]

                    # Sub-select single instance of collection for requested outputs
                    out = [dat[0] if ind in select else dat for ind, dat in enumerate(out)]

                else:
                    out = np.stack(outs)

            return out

        return wrapper

    return decorator
