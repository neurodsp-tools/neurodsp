"""Utility functions for filtering."""

import os
import json

import numpy as np
from scipy.signal import freqz, sosfreqz

from neurodsp.utils.decorators import multidim
from neurodsp.filt.checks import check_filter_definition

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
    f_range : tuple of (float, float), optional, default: False
        The lower and upper frequencies of the transition band.

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
    # This steps through the indices, in pairs, selecting from the vector to select from
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


def gen_filt_report(pass_type, filt_type, fs, f_db, db, pass_bw,
                    transition_bw, f_range, f_range_trans):
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

    Returns
    -------
    filt_report : dict
        A dicionary of filter parameter keys and corresponding values.
    """
    filt_report = {}

    # Filter type (high-pass, low-pass, band-pass, band-stop, FIR, IIR)
    filt_report['Pass Type'] = '{pass_type}'.format(pass_type=pass_type)

    # Cutoff frequenc(ies) (including definition)
    filt_report['Cutoff (Half-Amplitude)'] = '{cutoff} Hz'.format(cutoff=f_range)

    # Filter order (or length)
    filt_report['Filter Order'] = '{order}'.format(order=len(f_db)-1)

    # Roll-off or transition bandwidth
    filt_report['Transition Bandwidth'] = '{:.1f} Hz'.format(transition_bw)
    filt_report['Pass/Stop Bandwidth'] = '{:.1f} Hz'.format(pass_bw)

    # Passband ripple and stopband attenuation
    pb_ripple = np.max(db[:np.where(f_db < f_range_trans[0])[0][-1]])
    sb_atten = np.max(db[np.where(f_db > f_range_trans[1])[0][0]:])
    filt_report['Passband Ripple'] = '{pb_ripple} db'.format(pb_ripple=pb_ripple)
    filt_report['Stopband Attenuation'] = '{sb_atten} db'.format(sb_atten=sb_atten)

    # Filter delay (zero-phase, linear-phase, non-linear phase)
    filt_report['Filter Type'] = filt_type

    if filt_type == 'FIR':

        filt_report['Phase'] = '{filt_class}'.format(filt_class='linear-phase')
        filt_report['Group Delay'] = '{delay}s'.format(delay=(len(f_db)-1) / (2 * fs))
        filt_report['Direction'] = 'one-pass reverse'

    elif filt_type == 'IIR':

        # Group delay isn't reported for IIR since it varies from sample to sample
        filt_report['Phase'] = '{filt_class}'.format(filt_class='non-linear-phase')
        filt_report['Direction'] = 'two-pass forward and reverse'

    return filt_report


def save_filt_report(save_properties, filt_report):
    """Save filter properties as a json file.

    Parameters
    ----------
    save_properties : str
        Path, including file name, to save filter properites to as a json.
    filt_report : dict
        Contains filter report info.
    """

    # Ensure parents exists
    if not os.path.isdir(os.path.dirname(save_properties)):
        raise ValueError("Unable to save properties. Parent directory does not exist.")

    # Enforce file extension
    if not save_properties.endswith('.json'):
        save_properties = save_properties + '.json'

    # Save
    with open(save_properties, 'w') as file_path:
        json.dump(filt_report, file_path)
