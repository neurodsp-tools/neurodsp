"""Utility functions for filtering."""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz, sosfreqz

from neurodsp.utils.decorators import multidim
from neurodsp.filt.checks import check_filter_definition
from neurodsp.plts.filt import plot_frequency_response, plot_impulse_response

###################################################################################################
###################################################################################################

def infer_passtype(f_range):
    """Given frequency definition of a filter, infer the passtype.

    Parameters
    ----------
    f_range : tuple of (float, float)
        Cutoff frequency(ies) used for filter, specified as f_lo & f_hi.

    Returns
    -------
    pass_type : str
        Which kind of filter pass_type is consistent with the frequency definition provided.

    Notes
    -----
    Assumes that a definition with two frequencies is a 'bandpass' (not 'bandstop').
    """

    if f_range[0] is None:
        pass_type = 'lowpass'
    elif f_range[1] is None:
        pass_type = 'highpass'
    else:
        pass_type = 'bandpass'

    # Check the inferred passtype & frequency definition is valid
    _ = check_filter_definition(pass_type, f_range)

    return pass_type


def compute_frequency_response(filter_coefs, a_vals, fs):
    """Compute the frequency response of a filter.

    Parameters
    ----------
    filter_coefs : 1d or 2d array
        If 1d, interpreted as the B-value filter coefficients.
        If 2d, interpreted as the second-order (sos) filter coefficients.
    a_vals : 1d array or None
        The A-value filter coefficients for a filter.
        If second-order filter coefficients are provided in `filter_coefs`, must be None.
    fs : float
        Sampling rate, in Hz.

    Returns
    -------
    f_db : 1d array
        Frequency vector corresponding to attenuation decibels, in Hz.
    db : 1d array
        Degree of attenuation for each frequency specified in `f_db`, in dB.

    Examples
    --------
    Compute the frequency response for an FIR filter:

    >>> from neurodsp.filt.fir import design_fir_filter
    >>> filter_coefs = design_fir_filter(fs=500, pass_type='bandpass', f_range=(8, 12))
    >>> f_db, db = compute_frequency_response(filter_coefs, 1, fs=500)

    Compute the frequency response for an IIR filter, which uses SOS coefficients:

    >>> from neurodsp.filt.iir import design_iir_filter
    >>> sos_coefs = design_iir_filter(fs=500, pass_type='bandpass',
    ...                               f_range=(8, 12), butterworth_order=3)
    >>> f_db, db = compute_frequency_response(sos_coefs, None, fs=500)
    """

    if filter_coefs.ndim == 1 and a_vals is not None:
        # Compute response for B & A value filter coefficient inputs
        w_vals, h_vals = freqz(filter_coefs, a_vals, worN=int(fs * 2))
    elif filter_coefs.ndim == 2 and a_vals is None:
        # Compute response for sos filter coefficient inputs
        w_vals, h_vals = sosfreqz(filter_coefs, worN=int(fs * 2))
    else:
        raise ValueError("The organization of the filter coefficient inputs is not understood.")

    f_db = w_vals * fs / (2. * np.pi)
    db = 20 * np.log10(abs(h_vals))

    return f_db, db


def compute_pass_band(fs, pass_type, f_range):
    """Compute the pass bandwidth of a filter.

    Parameters
    ----------
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

    Returns
    -------
    pass_bw : float
        The pass bandwidth of the filter.

    Examples
    --------
    Compute the bandwidth of a bandpass filter:

    >>> compute_pass_band(fs=500, pass_type='bandpass', f_range=(5, 25))
    20.0
    """

    f_lo, f_hi = check_filter_definition(pass_type, f_range)
    if pass_type in ['bandpass', 'bandstop']:
        pass_bw = f_hi - f_lo
    elif pass_type == 'highpass':
        pass_bw = compute_nyquist(fs) - f_lo
    elif pass_type == 'lowpass':
        pass_bw = f_hi

    return pass_bw


def compute_transition_band(f_db, db, low=-20, high=-3, return_freqs=False):
    """Compute transition bandwidth of a filter.

    Parameters
    ----------
    f_db : 1d array
        Frequency vector corresponding to attenuation decibels, in Hz.
    db : 1d array
        Degree of attenuation for each frequency specified in `f_db`, in dB.
    low : float, optional, default: -20
        The lower limit that defines the transition band, in dB.
    high : float, optional, default: -3
        The upper limit that defines the transition band, in dB.
    return_freqs : bool, optional, default: False
        Whether to return a tuple of (lower, upper) frequency bounds for the transition band.

    Returns
    -------
    transition_band : float
        The transition bandwidth of the filter.
    f_range : tuple of (float, float)
        The lower and upper frequencies of the transition band.
        Only returned is return_freqs is True.

    Examples
    --------
    Compute the transition band of an FIR filter, using the computed frequency response:

    >>> from neurodsp.filt.fir import design_fir_filter
    >>> filter_coefs = design_fir_filter(fs=500, pass_type='bandpass', f_range=(1, 25))
    >>> f_db, db = compute_frequency_response(filter_coefs, 1, fs=500)
    >>> compute_transition_band(f_db, db, low=-20, high=-3)
    0.5

    Compute the transition band of an IIR filter, using the computed frequency response:

    >>> from neurodsp.filt.iir import design_iir_filter
    >>> sos = design_iir_filter(fs=500, pass_type='bandstop',
    ...                         f_range=(10, 20), butterworth_order=7)
    >>> f_db, db = compute_frequency_response(sos, None, fs=500)
    >>> compute_transition_band(f_db, db, low=-20, high=-3)
    2.0
    """

    # This gets the indices of transitions to the values in searched for range
    inds = np.where(np.diff(np.logical_and(db > low, db < high)))[0]

    # This determines at which frequencies the transition band occurs
    transition_pairs = [(a, b) for a, b in zip(f_db[inds[0::2]], f_db[inds[1::2]])]
    pair_idx = np.argmax([(tran[1] - tran[0]) for tran in transition_pairs])
    f_lo = transition_pairs[pair_idx][0]
    f_hi = transition_pairs[pair_idx][1]
    transition_band = f_hi - f_lo

    if return_freqs:

        return transition_band, (f_lo, f_hi)

    return transition_band


def compute_nyquist(fs):
    """Compute the Nyquist frequency.

    Parameters
    ----------
    fs : float
        Sampling rate, in Hz.

    Returns
    -------
    float
        The Nyquist frequency of a signal with the given sampling rate, in Hz.

    Examples
    --------
    Compute the Nyquist frequency for a 500 Hz sampling rate:

    >>> compute_nyquist(fs=500)
    250.0
    """

    return fs / 2.


@multidim()
def remove_filter_edges(sig, filt_len):
    """Drop the edges, by making NaN, from a filtered signal, to avoid edge artifacts.

    Parameters
    ----------
    sig : 1d array
        Filtered signal to have edge artifacts removed from.
    filt_len : int
        Length of the filter that was applied.

    Returns
    -------
    sig : 1d array
        Filter signal with edge artifacts switched to NaNs.

    Examples
    --------
    Apply a filter and remove the filter edges of the filtered signal:

    >>> from neurodsp.filt.fir import design_fir_filter, apply_fir_filter
    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> filter_coefs = design_fir_filter(fs=500, pass_type='bandpass', f_range=(1, 25))
    >>> filt_sig = apply_fir_filter(sig, filter_coefs)
    >>> filt_sig_no_edges = remove_filter_edges(filt_sig, filt_len=len(filter_coefs))
    """

    n_rmv = int(np.ceil(filt_len / 2))
    sig[:n_rmv] = np.nan
    sig[-n_rmv:] = np.nan

    return sig


def gen_filt_str(pass_type, filt_type, fs, f_db, db, pass_bw,
                 transition_bw, f_range, f_range_trans, order):
    """Create a filter report.

    Parameters
    ----------
    pass_type : {'bandpass', 'bandstop', 'lowpass', 'highpass'}
        Which type of filter was applied.
    filt_type : str, {'FIR', 'IIR'}
        The type of filter being applied.
    fs : float
        Sampling rate, in Hz.
    f_db : 1d array
        Frequency vector corresponding to attenuation decibels, in Hz.
    db : 1d array
        Degree of attenuation for each frequency specified in `f_db`, in dB.
    pass_bw : float
        The pass bandwidth of the filter.
    transition_band : float
        The transition bandwidth of the filter.
    f_range : tuple of (float, float) or float
        Cutoff frequency(ies) used for filter, specified as f_lo & f_hi.
    f_range_trans : tuple of (float, float)
        The lower and upper frequencies of the transition band.
    order : int
        The filter length for FIR filter or butterworth order for IIR filters.

    Returns
    -------
    filt_str : str
        Filter properties as a string that is ready to embed into a pdf report.
    """

    filt_str = []

    # Filter type (high-pass, low-pass, band-pass, band-stop, FIR, IIR)
    filt_str.append('Pass Type: {pass_type}'.format(pass_type=pass_type))

    # Cutoff frequenc(ies) (including definition)
    filt_str.append('Cutoff (Half-Amplitude): {cutoff} Hz'.format(cutoff=f_range))

    # Filter order (or length-1) for FIR or butterworth order for IIR
    filt_str.append('Filter Order: {order}'.format(order=order))

    # Roll-off or transition bandwidth
    filt_str.append('Transition Bandwidth: {:.1f} Hz'.format(transition_bw))
    filt_str.append('Pass/Stop Bandwidth: {:.1f} Hz'.format(pass_bw))

    # Passband ripple and stopband attenuation
    pb_ripple = np.max(db[:np.where(f_db < f_range_trans[0])[0][-1]])
    sb_atten = np.max(db[np.where(f_db > f_range_trans[1])[0][0]:])
    filt_str.append('Passband Ripple: {:1.4f} db'.format(pb_ripple))
    filt_str.append('Stopband Attenuation: {:1.4f} db'.format(sb_atten))

    # Filter delay (zero-phase, linear-phase, non-linear phase)
    filt_str.append('Filter Type: {filt_type}'.format(filt_type=filt_type))

    if filt_type == 'FIR':

        filt_str.append('Phase: linear-phase')
        filt_str.append('Group Delay: {:1.4f} s'.format((len(f_db)-1) / (2 * fs)))
        filt_str.append('Direction: one-pass reverse')

    elif filt_type == 'IIR':

        # Group delay isn't reported for IIR since it varies from sample to sample
        filt_str.append('Phase: non-linear-phase')
        filt_str.append('Direction: two-pass forward and reverse')

    # Format the list into a string
    filt_str = [

        # Header
        '=',
        '',
        'FILTER REPORT',
        '',

        # Settings
        *filt_str,

        # Footer
        '',
        '='
    ]

    str_len = 50
    filt_str [0] = filt_str [0] * str_len
    filt_str [-1] = filt_str [-1] * str_len

    filt_str  = '\n'.join([string.center(str_len) for string in filt_str])

    return filt_str


def save_filt_report(pdf_path, pass_type, filt_type, fs, f_db, db,  pass_bw, transition_bw,
                     f_range, f_range_trans, order, filter_coefs=None):
    """Save filter properties as a json file.

     Parameters
    ----------
    pdf_path: str
        Path, including file name, to save a filter report to as a pdf.
    pass_type : {'bandpass', 'bandstop', 'lowpass', 'highpass'}
        Which type of filter was applied.
    filt_type : str, {'FIR', 'IIR'}
        The type of filter being applied.
    fs : float
        Sampling rate, in Hz.
    f_db : 1d array
        Frequency vector corresponding to attenuation decibels, in Hz.
    db : 1d array
        Degree of attenuation for each frequency specified in `f_db`, in dB.
    pass_bw : float
        The pass bandwidth of the filter.
    transition_band : float
        The transition bandwidth of the filter.
    f_range : tuple of (float, float) or float
        Cutoff frequency(ies) used for filter, specified as f_lo & f_hi.
    f_range_trans : tuple of (float, float)
        The lower and upper frequencies of the transition band.
    order : int
        The filter length for FIR filter or butterworth order for IIR filters.
    filter_coefs : 1d array, optional, default: None
        Filter coefficients of the FIR filter.
    """

    # Ensure valid path
    if not pdf_path.startswith('/') and not pdf_path.startswith('./'):
        pdf_path = './' + pdf_path

    if not os.path.isdir(os.path.dirname(pdf_path)):
        raise ValueError("Unable to save properties. Parent directory does not exist.")

    # Enforce file extension
    if not pdf_path.endswith('.pdf'):
        pdf_path = pdf_path + '.pdf'

    # Create properties string
    filt_str = gen_filt_str(pass_type, filt_type, fs, f_db, db, pass_bw,
                            transition_bw, f_range, f_range_trans, order)

    # Plot
    if filter_coefs is not None:

        _, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 18),
                                 gridspec_kw={'height_ratios': [1, 4, 4]})

        # Plot impulse response for IIR filters
        plot_impulse_response(fs, filter_coefs, ax=axes[2])

    else:

        _, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10),
                                 gridspec_kw={'height_ratios': [1, 4]})

    # Plot filter parameter string
    font = {'family': 'monospace', 'weight': 'normal', 'size': 16}
    axes[0].text(0.5, 0.7, filt_str, font, ha='center', va='center')
    axes[0].set_frame_on(False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plot filter responses
    plot_frequency_response(f_db, db, ax=axes[1])

    # Save
    plt.savefig(pdf_path)
    plt.close()
