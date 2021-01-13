"""Checker functions for filtering."""

from warnings import warn

import numpy as np

###################################################################################################
###################################################################################################

def check_filter_definition(pass_type, f_range):
    """Check a filter definition for validity, and get filter frequency range of the filter.

    Parameters
    ----------
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
    f_lo : float or None
        The lower frequency range of the filter, specifying the highpass frequency, if specified.
    f_hi : float or None
        The higher frequency range of the filter, specifying the lowpass frequency, if specified.

    Raises
    ------
    ValueError
        If the filter passtype it not understood, or cutoff frequencies are incompatible.

    Examples
    --------
    Check a filter definition, and get the cutoff frequencies returned, if the filter is valid:

    >>> f_hi, f_lo = check_filter_definition(pass_type='bandpass', f_range=(5, 25))

    Check a filter definition, for an invalid filter.
    This example fails since a bandpass filter requires two values for ``f_range``.

    >>> try:
    ...     f_hi, f_lo = check_filter_definition(pass_type='bandpass', f_range=(20))
    ... except ValueError:
    ...     print("The filter definition is invalid.")
    The filter definition is invalid.
    """

    if pass_type not in ['bandpass', 'bandstop', 'lowpass', 'highpass']:
        raise ValueError('Filter passtype not understood.')

    # Check that frequency cutoff inputs are appropriate
    #   For band filters, 2 inputs required & second entry must be > first
    if pass_type in ('bandpass', 'bandstop'):
        if isinstance(f_range, (tuple, list)) and f_range[0] >= f_range[1]:
            raise ValueError('Second cutoff frequency must be greater than first.')
        elif isinstance(f_range, (int, float)) or len(f_range) != 2:
            raise ValueError('Two cutoff frequencies required for bandpass and bandstop filters.')

        # Map f_range to f_lo and f_hi
        f_lo, f_hi = f_range

    # For lowpass and highpass can be tuple or int/float
    if pass_type == 'lowpass':
        if isinstance(f_range, (int, float)):
            f_hi = f_range
        elif isinstance(f_range, (tuple, list)):
            f_hi = f_range[1]
        f_lo = None

    if pass_type == 'highpass':
        if isinstance(f_range, (int, float)):
            f_lo = f_range
        elif isinstance(f_range, (tuple, list)):
            f_lo = f_range[0]
        f_hi = None

    # Make sure pass freqs are floats
    f_lo = float(f_lo) if f_lo else f_lo
    f_hi = float(f_hi) if f_hi else f_hi

    return f_lo, f_hi


def check_filter_properties(filter_coefs, a_vals, fs, pass_type, f_range,
                            transitions=(-20, -3), filt_type=None, verbose=True):
    """Check a filters properties, including pass band and transition band.

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
    transitions : tuple of (float, float), optional, default: (-20, -3)
        Cutoffs, in dB, that define the transition band.
    verbose : bool, optional, default: True
        Whether to print out transition and pass bands.

    Returns
    -------
    passes : bool
        Whether all the checks pass. False if one or more checks fail.

    Examples
    --------
    Check the properties of an FIR filter:

    >>> from neurodsp.filt.fir import design_fir_filter
    >>> fs, pass_type, f_range = 500, 'bandpass', (1, 25)
    >>> filter_coefs = design_fir_filter(fs, pass_type, f_range)
    >>> passes = check_filter_properties(filter_coefs, 1, fs, pass_type, f_range)
    Transition bandwidth is 0.5 Hz.
    Pass/stop bandwidth is 24.0 Hz.
    """

    # Import utility functions inside function to avoid circular imports
    from neurodsp.filt.utils import (compute_frequency_response,
                                     compute_pass_band, compute_transition_band)

    # Initialize variable to keep track if all checks pass
    passes = True

    # Compute the frequency response
    f_db, db = compute_frequency_response(filter_coefs, a_vals, fs)

    # Check that frequency response goes below transition level (has significant attenuation)
    if np.min(db) >= transitions[0]:
        passes = False
        warn('The filter attenuation never goes below {} dB.'\
             'Increase filter length.'.format(transitions[0]))
        # If there is no attenuation, cannot calculate bands, so return here
        return passes

    # Check that both sides of a bandpass have significant attenuation
    if pass_type == 'bandpass':
        if db[0] >= transitions[0] or db[-1] >= transitions[0]:
            passes = False
            warn('The low or high frequency stopband never gets attenuated by'\
                 'more than {} dB. Increase filter length.'.format(abs(transitions[0])))

    # Compute pass & transition bandwidth
    pass_bw = compute_pass_band(fs, pass_type, f_range)
    transition_bw, f_range_trans = compute_transition_band(f_db, db, transitions[0], transitions[1],
                                                           return_freqs=True)

    # Raise warning if transition bandwidth is too high
    if transition_bw > pass_bw:
        passes = False
        warn('Transition bandwidth is  {:.1f}  Hz. This is greater than the desired'\
             'pass/stop bandwidth of  {:.1f} Hz'.format(transition_bw, pass_bw))

    # Print out transition bandwidth and pass bandwidth to the user
    if verbose:

        # Filter type (high-pass, low-pass, band-pass, band-stop, FIR, IIR)
        print('Pass Type: {pass_type}'.format(pass_type=pass_type))

        # Cutoff frequency (including definition)
        cutoff = round(np.min(f_range) + (0.5 * transition_bw), 3)
        print('Cutoff (half-amplitude): {cutoff} Hz'.format(cutoff=cutoff))

        # Filter order (or length)
        print('Filter order: {order}'.format(order=len(f_db)-1))

        # Roll-off or transition bandwidth
        print('Transition bandwidth: {:.1f} Hz'.format(transition_bw))
        print('Pass/stop bandwidth: {:.1f} Hz'.format(pass_bw))

        # Passband ripple and stopband attenuation
        pb_ripple = np.max(db[:np.where(f_db < f_range_trans[0])[0][-1]])
        sb_atten = np.max(db[np.where(f_db > f_range_trans[1])[0][0]:])
        print('Passband Ripple: {pb_ripple} db'.format(pb_ripple=pb_ripple))
        print('Stopband Attenuation: {sb_atten} db'.format(sb_atten=sb_atten))

        # Filter delay (zero-phase, linear-phase, non-linear phase)
        if filt_type == 'FIR' and pass_type in ['bandstop', 'lowpass']:
            filt_class = 'linear-phase'
        elif filt_type == 'FIR' and pass_type in ['bandpass', 'highpass']:
            filt_class = 'zero-phase'
        elif filt_type == 'IIR':
            filt_class = 'non-linear-phase'
        else:
            filt_class = None

        if filt_type is not None:
            print('Filter Class: {filt_class}'.format(filt_class=filt_class))

        if filt_class == 'linear-phase':
            print('Group Delay: {delay}s'.format(delay=(len(f_db)-1) / 2 * fs))
        elif filt_class == 'zero-phase':
            print('Group Delay: 0s')

        # Direction of computation (one-pass forward/reverse, or two-pass forward and reverse)
        if filt_type == 'FIR':
            print('Direction: one-pass reverse.')
        else:
            print('Direction: two-pass forward and reverse')

    return passes


def check_filter_length(sig_length, filt_len):
    """Check the length of a filter against the length of a signal.

    Parameters
    ----------
    sig_length : int
        The length of the signal to be filtered.
    filt_len : int
        The length of the filter.

    Raises
    ------
    ValueError
        If the signal length is too short for the filter length.
    """

    if filt_len >= sig_length:
        raise ValueError(
            'The designed filter (length: {:d}) is longer than the signal '\
            '(length: {:d}). The filter needs to be shortened by decreasing '\
            'the n_cycles or n_seconds parameter. However, this will decrease '\
            'the frequency resolution of the filter.'.format(filt_len, sig_length))
