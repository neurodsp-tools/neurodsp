"""Decorators."""

from functools import wraps

import numpy as np

from neurodsp.utils.norm import normalize_sig

###################################################################################################
###################################################################################################

def normalize(func, **kwargs):
    """Decorator function to normalize the first output of the wrapped function.

    Notes
    -----
    If shift or scale keyword arguments are passed, signal normalization will be bypassed.
    """

    @wraps(func)
    def decorated(*args, **kwargs):

        # Grab variance & mean as possible kwargs, with default values if not
        mean = kwargs.pop('mean', 0.)
        variance = kwargs.pop('variance', 1.)

        # Call sim function, and unpack to get sig variable, if there are multiple returns
        out = func(*args, **kwargs)
        sig = out[0] if isinstance(out, tuple) else out

        # Skip normalization if scale or shift is passed
        if 'scale' in kwargs.keys() or 'shift' in kwargs.keys():
            return sig

        # Normalize signal, applying mean and variance transformations
        sig = normalize_sig(sig, mean, variance)

        # Return sig & other outputs, if there were any, or just sig otherwise
        return (sig, out[1:]) if isinstance(out, tuple) else sig

    return decorated


def multidim(select=[], pass_2d_input=False):
    """Decorator function to apply the wrapped function across dimensions.

    Parameters
    ----------
    select : list of int, optional
        List of indices of outputs to sub-select a single instance from.
    pass_2d_input : bool, optional, default: False
        If True, passes 2d arrays to function being wrapped.

    Notes
    -----
    This decorator assumes the wrapped function has the data input 'sig' as the first argument.
    """

    def decorator(func, *args, **kwargs):

        @wraps(func)
        def wrapper(sig, *args, **kwargs):

            if sig.ndim == 1:
                out = func(sig, *args, **kwargs)
            elif sig.ndim == 2 and not pass_2d_input:

                # Apply func across rows of the input data
                outs = [func(data, *args, **kwargs) for data in sig]

                if isinstance(outs[0], tuple):

                    # Collect associated outputs from each, in case there are multiple outputs
                    out = [np.stack([data[n_out] for data in outs]) \
                        for n_out in range(len(outs[0]))]

                    # Sub-select single instance of collection for requested outputs
                    out = [data[0] if ind in select else data for ind, data in enumerate(out)]

                else:
                    out = np.array(outs)

            else:
                # Reshape to 2d and run func
                shape = sig.shape
                sig_2d = sig.reshape(-1, shape[-1])
                if pass_2d_input:
                    out = func(sig_2d, *args, **kwargs)
                else:
                    out = wrapper(sig_2d, *args, **kwargs)

                # Reshape back to original shape
                if isinstance(out, (tuple, list)):
                    for ind in range(len(out)):
                        if ind not in select:
                            out[ind] = out[ind].reshape((*shape[:-1], -1))
                            if out[ind].shape[-1] == 1:
                                # Last dim is extraneous (e.g. func returns a scalar) so squeeze
                                out[ind] = out[ind].reshape(list(out[ind].shape)[:-1])
                else:
                    out = out.reshape((*shape[:-1], -1))

                    if out.shape[-1] == 1:
                        # Last dim is extraneous (e.g. func returns a scalar) so squeeze
                        out = out.reshape(list(out.shape)[:-1])

            return out

        return wrapper

    return decorator
