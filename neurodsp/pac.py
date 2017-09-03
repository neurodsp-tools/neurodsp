"""
pac.py
Compute the phase-amplitude coupling between two oscillators
"""
import numpy as np
import neurodsp
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm


def compute_pac(x_pha, x_amp, Fs, f_range_lo, f_range_hi,
                N_seconds_lo=None, N_seconds_hi=None,
                filter_fn=None, filter_kwargs=None,
                hilbert_increase_N=False,
                pac_method='ozkurt',
                N_bins_tort=None, N_surr_canolty=None, verbose=True):
    """
    Calculate phase-amplitude coupling between a low-frequency
    range of x_pha and a higher frequency range in x_amp

    Parameters
    ----------
    x_pha : array-like, 1d
        The time-series from which to compute the phase component
    x_amp : array-like, 1d
        The time series from which to compute the amplitude component
    Fs : float
        Sampling rate (Hz) of the two time series
    f_range_lo : tuple, 2 elements
        The low frequency filtering range (Hz)
    f_range_hi : tuple, 2 elements
        The high frequency filtering range (Hz)
    N_seconds_lo : float
        Length of the low band-pass filter (seconds)
    N_seconds_hi : float
        Length of the high band-pass filter (seconds)
    filter_fn : function or False
        The filtering function, with api:
        `filterfn(x, Fs, pass_type, f_lo, f_hi, remove_edge_artifacts=True)
        If False, it is assumed that x_pha and x_amp are the phase time
        series and the amplitude time series, respectively. Therefore, no
        filtering or hilbert transform will be done.
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
    hilbert_increase_N : bool
        if True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because scipy.signal.hilbert can be very slow for some lengths of x
    pac_method : str
        Indicates the method used to correlate the phase and amplitude time series in order to
        quantify the strength of pac.
        'plv': phase-locking value method (see Penny et al., 2008, J Neuro Methods)
        'glm': general linear model method (see Penny et al., 2008, J Neuro Methods)
        'tort': modulation index method (see Tort et al., 2010, J Neurophys)
        'canolty' : modulation index method (see Canolty et al., 2006, Science)
        'ozkurt' : normalized modulation index method (see Ozkurt & Schnitzler, 2011, J Neuro Methods)
    N_bins_tort : int or None
        Number of phase bins to use in Tort's modulation index method of estimating PAC
    N_surr_canolty : int or None
        Number of surrograte runs for Canolty's modulation index method of estimating PAC
    verbose : bool
        if True, print optional warning information

    Returns
    -------
    pac : float
        phase-amplitude coupling strength
    """
    # Set default filtering parameters
    if N_seconds_lo is None:
        if verbose:
            warnings.warn('Filter order not specified. Filter length automatically set to 3 cycles of the low cutoff frequency.')
        N_cycles = 3
        N_seconds_lo = N_cycles / f_range_lo[0]
    if N_seconds_hi is None:
        if verbose:
            warnings.warn('Filter order not specified. Filter length automatically set to 3 cycles of the low cutoff frequency.')
        N_cycles = 3
        N_seconds_hi = N_cycles / f_range_hi[0]
    if filter_fn is None:
        filter_fn = neurodsp.filter
    if filter_kwargs is None:
        filter_kwargs = {}

    # Only compute phase and amplitude if filter_fn is not False
    if filter_fn is not False:
        # Compute phase time series
        filter_kwargs['N_seconds'] = N_seconds_lo
        filter_kwargs['verbose'] = verbose
        pha = neurodsp.phase_by_time(x_pha, Fs, f_range_lo,
                                     filter_fn=filter_fn, filter_kwargs=filter_kwargs,
                                     hilbert_increase_N=hilbert_increase_N)

        # Compute amp time series
        filter_kwargs['N_seconds'] = N_seconds_hi
        amp = neurodsp.amp_by_time(x_amp, Fs, f_range_hi,
                                   filter_fn=filter_fn, filter_kwargs=filter_kwargs,
                                   hilbert_increase_N=hilbert_increase_N)
    else:
        # Set phase and amplitude time series to 'x' if filter_fn set to False
        pha = x_pha
        amp = x_amp

        # Reset filter function and kwargs
        filter_fn = neurodsp.filter
        filter_kwargs = {'verbose': verbose}

    # Remove the part of both signals with edge artifacts
    # The filter should be longer for the lower-frequency phase-providing
    # signal
    first_nonan = np.where(~np.isnan(pha))[0][0]
    last_nonan = np.where(~np.isnan(pha))[0][-1] + 1
    pha_nonan = pha[first_nonan:last_nonan]
    amp_nonan = amp[first_nonan:last_nonan]

    # Compute statistic relating phase and amplitude
    if pac_method == 'plv':
        pac = _plv_pac(pha, amp, Fs, f_range_lo, N_seconds_lo,
                       filter_fn, filter_kwargs, hilbert_increase_N)
    elif pac_method == 'glm':
        pac = _glm_pac(pha, amp)
    elif pac_method == 'tort':
        pac = _tort_pac(pha, amp, N_bins_tort)
    elif pac_method == 'canolty':
        pac = _canolty_pac(pha, amp, N_surr_canolty)
    elif pac_method == 'ozkurt':
        pac = _ozkurt_pac(pha, amp)
    else:
        raise ValueError('Method specified in "pac_method" not known.')
    return pac


def _plv_pac(pha, amp, Fs, f_range_lo, N_seconds_lo,
             filter_fn, filter_kwargs, hilbert_increase_N):
    """Use the PLV method to compute phase-amplitude coupling"""

    # Compute the phase of the amplitude time series
    filter_kwargs['N_seconds'] = N_seconds_lo
    amp_pha = neurodsp.phase_by_time(amp, Fs, f_range_lo,
                                     filter_fn=filter_fn, filter_kwargs=filter_kwargs,
                                     hilbert_increase_N=hilbert_increase_N)

    # Compute phase locking
    pac = np.abs(np.mean(np.exp(1j * (pha - amp_pha))))
    return pac


def _tort_pac(pha, amp, N_bins=None):
    """Use Tort's modulation index method to compute phase-amplitude coupling"""
    # Set default bin number
    if N_bins is None:
        N_bins = 20

    # Convert the phase time series from radians to degrees
    phadeg = np.degrees(pha)

    # Calculate mean amplitude in each phase bin
    binsize = 360 / N_bins
    phase_lo = np.arange(-180, 180, binsize)
    mean_amp = np.zeros(len(phase_lo))
    for b in range(len(phase_lo)):
        phaserange = np.logical_and(phadeg >= phase_lo[b],
                                    phadeg < (phase_lo[b] + binsize))
        mean_amp[b] = np.mean(amp[phaserange])

    # Compute the probability of an amplitude unit being in a phase bin
    p_j = mean_amp / sum(mean_amp)

    # Compute the entropy and PAC
    h = -np.sum(p_j * np.log10(p_j))
    h_max = np.log10(N_bins)
    pac = (h_max - h) / h_max

    return pac


def _glm_pac(pha, amp):
    """Use the GLM method to compute phase-amplitude coupling"""
    # Prepare GLM
    y = amp
    X_pre = np.vstack((np.cos(pha), np.sin(pha)))
    X = X_pre.T

    # Apply GLM and compute residuals
    y_hat, beta_hat = _ols(y, X)
    resid = y - y_hat

    # Calculate PAC from GLM residuals
    pac = 1 - np.sum(resid ** 2) / np.sum((amp - np.mean(amp)) ** 2)
    return pac


def _ols(y, X):
    """Custom OLS (to minimize outside dependecies)"""
    dummy = np.repeat(1.0, X.shape[0])
    X = np.hstack([X, dummy[:, np.newaxis]])
    beta_hat, resid, _, _ = np.linalg.lstsq(X, y)
    y_hat = np.dot(X, beta_hat)
    return y_hat, beta_hat


def _canolty_pac(pha, amp, N_surr=None):
    """Use Canolty's modulation index method to compute phase-amplitude coupling"""
    # Set default number of surrogate runs
    if N_surr is None:
        N_surr = 100

    # Calculate modulation index
    pac_raw = np.abs(np.mean(amp * np.exp(1j * pha)))

    # Calculate surrogate MIs
    pacS = np.zeros(N_surr)
    loj = np.exp(1j * pha)
    for s in range(N_surr):
        loS = np.roll(loj, np.random.randint(len(pha)))
        pacS[s] = np.abs(np.mean(amp * loS))

    # Return z-score of observed modulation index compared to null distribution
    pac = (pac_raw - np.mean(pacS)) / np.std(pacS)
    return pac


def _ozkurt_pac(pha, amp):
    """Use Ozkurt's method to compute phase-amplitude coupling"""
    # Calculate normalized modulation index
    pac = np.abs(np.sum(amp * np.exp(1j * pha))) / \
        (np.sqrt(len(pha)) * np.sqrt(np.sum(amp**2)))
    return pac


def compute_pac_comodulogram(x_pha, x_amp, Fs,
                             f_pha_bin_edges, f_amp_bin_edges,
                             N_cycles_pha=None, N_cycles_amp=None,
                             filter_fn=None, filter_kwargs=None,
                             hilbert_increase_N=False,
                             pac_method='ozkurt',
                             N_bins_tort=None, N_surr_canolty=None, verbose=True):
    """
    Calculate phase-amplitude coupling between a low-frequency
    range of x_pha and a higher frequency range in x_amp

    Parameters
    ----------
    x_pha : array-like, 1d
        The time-series from which to compute the phase component
    x_amp : array-like, 1d
        The time series from which to compute the amplitude component
    Fs : float
        Sampling rate (Hz) of the two time series
    f_pha_bin_edges: array-like, 1d
        An array of frequency values (Hz) that define the edges of the
        frequency ranges on which to estimate phase
    f_amp_bin_edges : array-like, 1d
        An array of frequency values (Hz) that define the edges of the
        frequency ranges on which to estimate amplitude
    N_cycles_pha : float
        Length of the low band-pass filter in terms of the number of cycles
        of a sine wave with a frequency at the low-cutoff of the bandpass filter
    N_cycles_amp : float
        Length of the high band-pass filter in terms of the number of cycles
        of a sine wave with a frequency at the low-cutoff of the bandpass filter
    filter_fn : function or False
        The filtering function, with api:
        `filterfn(x, Fs, pass_type, f_lo, f_hi, remove_edge_artifacts=True)
        If False, it is assumed that x_pha and x_amp are the phase time
        series and the amplitude time series, respectively. Therefore, no
        filtering or hilbert transform will be done.
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
    hilbert_increase_N : bool
        if True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because scipy.signal.hilbert can be very slow for some lengths of x
    pac_method : str
        Indicates the method used to correlate the phase and amplitude time series in order to
        quantify the strength of pac.
        'plv': phase-locking value method (see Penny et al., 2008, J Neuro Methods)
        'glm': general linear model method (see Penny et al., 2008, J Neuro Methods)
        'tort': modulation index method (see Tort et al., 2010, J Neurophys)
        'canolty' : modulation index method (see Canolty et al., 2006, Science)
        'ozkurt' : normalized modulation index method (see Ozkurt & Schnitzler, 2011, J Neuro Methods)
    N_bins_tort : int or None
        Number of phase bins to use in Tort's modulation index method of estimating PAC
    N_surr_canolty : int or None
        Number of surrograte runs for Canolty's modulation index method of estimating PAC
    verbose : bool
        if True, print optional warning information

    Returns
    -------
    pac : 2d array
        phase-amplitude coupling strength values for each combination of phase-providing
        frequency bin and amplitude-providing frequency bin.
    """

    # Display warning about the true width of frequency bins
    if verbose:
        warnings.warn("The true bandwidth of the filters used for each frequency bin of the comodulogram "
                      "is almost always are wider than the declared width of the frequency bin. "
                      "And this width increases as a function of frequency."
                      "For example the frequency bin 60-64Hz likely uses a bandwidth >4Hz. "
                      "You can decrease this bandwidth by increasing the N_cycles_pha and N_cycles_amp arguments. "
                      "This warning can be turned off by setting the 'verbose' kwarg to False.")

    # Set default filtering parameters
    if filter_fn is None:
        filter_fn = neurodsp.filter
    if filter_kwargs is None:
        filter_kwargs = {'N_cycles': N_cycles_pha,
                         'verbose': False}
    else:
        filter_kwargs['N_cycles'] = N_cycles_pha
        filter_kwargs['verbose'] = False

    # Compute phase time series for each frequency bin
    N_bins_pha = len(f_pha_bin_edges) - 1
    pha_by_bin = np.zeros((N_bins_pha, len(x_pha)))
    for i in range(N_bins_pha):
        f_range_temp = (f_pha_bin_edges[i], f_pha_bin_edges[i + 1])
        pha_by_bin[i] = neurodsp.phase_by_time(x_pha, Fs, f_range_temp,
                                               filter_fn=filter_fn,
                                               filter_kwargs=filter_kwargs,
                                               hilbert_increase_N=False)

    # Compute amplitude time series for each frequency bin
    N_bins_amp = len(f_amp_bin_edges) - 1
    amp_by_bin = np.zeros((N_bins_amp, len(x_pha)))
    for i in range(N_bins_amp):
        f_range_temp = (f_amp_bin_edges[i], f_amp_bin_edges[i + 1])
        amp_by_bin[i] = neurodsp.amp_by_time(x_amp, Fs, f_range_temp,
                                             filter_fn=filter_fn,
                                             filter_kwargs=filter_kwargs,
                                             hilbert_increase_N=False)

    # For each pair of frequency bins, compute PAC
    pac = np.zeros((N_bins_pha, N_bins_amp))
    for i in range(N_bins_pha):
        for j in range(N_bins_amp):
            f_range_pha_temp = (f_pha_bin_edges[i], f_pha_bin_edges[i + 1])
            f_range_amp_temp = (f_amp_bin_edges[j], f_amp_bin_edges[j + 1])
            pac[i, j] = compute_pac(pha_by_bin[i], amp_by_bin[j], Fs,
                                    f_range_pha_temp, f_range_amp_temp,
                                    filter_fn=False, pac_method=pac_method,
                                    N_bins_tort=N_bins_tort, N_surr_canolty=N_surr_canolty,
                                    verbose=False)
    return pac


def plot_pac_comodulogram(pac, f_pha_bin_edges, f_amp_bin_edges,
                          clim=None, figsize=None, colormap=None):
    """
    Plot the PAC comodulogram computed in the `compute_pac_comodulogram` function

    Parameters
    ----------
    pac : 2d array
        phase-amplitude coupling strength values for each combination of phase-providing
        frequency bin and amplitude-providing frequency bin.
    f_pha_bin_edges: array-like, 1d
        An array of frequency values (Hz) that define the edges of the
        frequency ranges on which to estimate phase
    f_amp_bin_edges : array-like, 1d
        An array of frequency values (Hz) that define the edges of the
        frequency ranges on which to estimate amplitude
    clim : 2-element tuple
        Limits in the colorbar that represents PAC strength
    figsize: 2-element tuple
        size of figure, as in the 'figsize' kwarg used in plt.figure()
    colormap: matplotlib-compatible colormap
        a colormap from matplotlib's colormap module (matplotlib.cm)
    """

    # Set defaults
    if figsize is None:
        figsize = (6, 5)
    if colormap is None:
        colormap = cm.viridis

    # Plot comodulogram
    plt.figure(figsize=figsize)
    # Set up colorbar
    if clim is None:
        cax = plt.pcolor(f_pha_bin_edges, f_amp_bin_edges, pac.T,
                         cmap=colormap)
        cbar = plt.colorbar(cax)
    else:
        cax = plt.pcolor(f_pha_bin_edges, f_amp_bin_edges, pac.T,
                         cmap=colormap, vmin=clim[0], vmax=clim[1])
        cbar = plt.colorbar(cax, ticks=clim)
        cbar.ax.set_yticklabels(clim, size=20)
    cbar.ax.set_ylabel('Modulation Index', size=20)

    # Plot labels
    plt.axis([f_pha_bin_edges[0], f_pha_bin_edges[-1],
              f_amp_bin_edges[0], f_amp_bin_edges[-1]])
    plt.xlabel('Phase frequency (Hz)', size=20)
    plt.ylabel('Amplitude frequency (Hz)', size=20)
    plt.yticks(size=20)
    plt.xticks(size=20)
    plt.tight_layout()
