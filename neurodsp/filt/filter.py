"""Filter time series."""

from neurodsp.filt.fir import filter_signal_fir
from neurodsp.filt.iir import filter_signal_iir
from neurodsp.utils.checks import check_param_options
from neurodsp.utils.decorators import multidim

###################################################################################################
###################################################################################################

@multidim(pass_2d_input=True)
def filter_signal(sig, fs, pass_type, f_range, filter_type=None,
                  print_transitions=False, plot_properties=False, return_filter=False,
                  **filter_kwargs):
    """Apply a bandpass, bandstop, highpass, or lowpass filter to a neural signal.

    Parameters
    ----------
    sig : array
        Time series to be filtered.
    fs : float
        Sampling rate, in Hz.
    pass_type : {'bandpass', 'bandstop', 'lowpass', 'highpass'}
        Which kind of filter to apply:

        * 'bandpass': apply a bandpass filter
        * 'bandstop': apply a bandstop (notch) filter
        * 'lowpass': apply a lowpass filter
        * 'highpass' : apply a highpass filter
    f_range : tuple of (float, float) or float
        Cutoff frequency(ies) used for filter, specified as f_lo & f_hi.
        For 'bandpass' & 'bandstop', must be a tuple.
        For 'lowpass' or 'highpass', can be a float that specifies pass frequency, or can be
        a tuple and is assumed to be (None, f_hi) for 'lowpass', and (f_lo, None) for 'highpass'.
    filter_type : {'fir', 'iir'}, optional
        Whether to use an FIR or IIR filter. IIR option is a butterworth filter.
        If None, type is inferred from input parameters, and/or defaults to FIR.
    print_transitions : bool, optional, default: True
        If True, print out the transition and pass bandwidths.
    plot_properties : bool, optional, default: False
        If True, plot the properties of the filter, including frequency response and/or kernel.
    return_filter : bool, optional, default: False
        If True, return the filter coefficients.
    **filter_kwargs
        Additional parameters for the filtering function, specific to filtering type.

        | For FIR filters, can include:
        |    n_cycles : float, optional
        |        Filter length, in number of cycles, defined at 'f_lo' frequency.
        |        Either `n_cycles` or `n_seconds` can be set for the filter length, but not both.
        |        If not provided, and `n_seconds` is also not defined, defaults to 3.
        |    n_seconds : float, optional
        |        Filter length, in seconds.
        |        Either `n_cycles` or `n_seconds` can be set for the filter length, but not both.
        |    remove_edges : bool, optional, default: True
        |        If True, replace samples within half the kernel length to be np.nan.
        | For IIR filters, can include:
        |    butterworth_order : int, optional
        |        Order of the butterworth filter. See input 'N' in scipy.signal.butter.

    Returns
    -------
    sig_filt : 1d array
        Filtered time series.
    kernel : 1d array or tuple of (1d array, 1d array)
        Filter coefficients. Only returned if `return_filter` is True.

    Examples
    --------
    Apply an FIR band pass filter to a signal, for the range of 1 to 25 Hz:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> filt_sig = filter_signal(sig, fs=500, pass_type='bandpass',
    ...                          filter_type='fir', f_range=(1, 25))
    """

    if filter_type is not None:
        check_param_options(filter_type, 'filter_type', ['fir', 'iir'])
    else:
        # Infer IIR if relevant parameter set, otherwise, assume FIR
        filter_type = 'iir' if 'butterworth_order' in filter_kwargs else 'fir'

    _filter_input_checks(filter_type, filter_kwargs)

    if filter_type.lower() == 'fir':
        return filter_signal_fir(sig, fs, pass_type, f_range, **filter_kwargs,
                                 print_transitions=print_transitions,
                                 plot_properties=plot_properties,
                                 return_filter=return_filter)

    elif filter_type.lower() == 'iir':
        return filter_signal_iir(sig, fs, pass_type, f_range, **filter_kwargs,
                                 print_transitions=print_transitions,
                                 plot_properties=plot_properties,
                                 return_filter=return_filter)


FILTER_INPUTS = {
    'fir' : ['n_cycles', 'n_seconds', 'remove_edges'],
    'iir' : ['butterworth_order'],
}


def _filter_input_checks(filter_type, filter_kwargs):
    """Check inputs to `filter_signal` match filter type."""

    for param in filter_kwargs.keys():
        assert param in FILTER_INPUTS[filter_type], \
            'Parameter {} not expected for {} filter'.format(param, filter_type)
