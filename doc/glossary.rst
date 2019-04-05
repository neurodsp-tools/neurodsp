NeuroDSP Glossary
=================

The following is a glossary of neuroscience and digital processing related terms that are used in NeuroDSP.

General
-------
.. glossary::

    time domain
        Signals that are represented as variations over time, and analyses of such signals.
    frequency domain
        Signals that are represented in terms of frequencies, and Analyses of such signals.
    sampling rate
        The rate at which samples are taken.
    periodic
        Properties or components of a signal that are rhythmic.
    aperiodic
        Properties or components of a signal that are arrhythmic, with no characteristic frequency.
    temporal resolution
        The precision of a measurement, in the time domain.
    frequency resolution
        The precision of a measurement, in the frequency domain.

Units
-----
.. glossary::

    Hertz (Hz)
        A unit of frequency, as the number of cycles per second.
    Decibels (dB)
        A unit of intensity, on a logarithmic scale.
    Volts (V)
        A unit of voltage, typically in the microvolt (uV) range for neural time series.

Filters
-------
.. glossary::

    Impulse Response
        The response of a fitler when presented with an impulse; a single, brief input.
    FIR
        A Finite Impulse Response filter, meaning the response to a single input is finite, settling to zero.
    IIR
        An Infinite Impulse Response filter, meaning the response to a single input continue infinitely.
    passband
        The range (band) of frequencies that can pass through a filter.
    stopband
        The range (band) of frequencies that are attenuated (stopped) by a filter.
    passtype
        The type of filter, defined in terms of what frequency bands or ranges it passes through, or filters out.

        * bandpass: a filter whose passband is a specific frequency band, bound by a low and high frequency point.
        * bandstop: a filter that passes through all frequencies except a band region that is attenuated.
        * lowpass: a filter whose passband is all frequencies below a filter frequency (low frequencies pass through).
        * highpass: a filter whose passband is all frequencies above a filter frequency (high frequencies pass through).
    transition band
        The range of frequencies that are in the transition region between the passband and the stopband.
    frequency response
        The response profile of a filter, specifying the gain and phase shift applied by the filter at each frequency.

Rhythms & Bursts
----------------
.. glossary::

    burst
        Periodic activity that lasts for a finite time , as in a 'burst of oscillatory activity'.

Time Frequency
--------------
.. glossary::

    frequency
        The number of occurences over a unit of time.
    phase
        The position, at a point in time, on a waveform cycle.
    amplitude
        The magnitude of a signal, as the peak-to-trough.
    period
        A single cycle of a rhythm, defined as the time between two consecutive troughs (or peaks).
    hilbert transform
        A mathematical transform that derives the analyic representation of a signal, where the
        analytic representation is complex-valued representation that can be used to find the
        analytic amplitude and phase of a signal.
    wavelet
        A wave-like signal, or 'brief oscillation', that starts at zero amplitude, increases
        in amplitude to some value, and then decays back to zero.

Spectral
--------
.. glossary::

    fourier transform
        A mathematical transformation to decompose a time series into its constituent frequencies.
    power spectrum
        A frequency domain representation, as an estimate of the power across frequencies in a signal.
    median filter
        A smoothing approach to replace each value in a signal with the median of the neighbouring entries.
    coefficient of variation
        A standardized measure of dispersion, as the ratio of the standard deviation to the mean.

Simulations
-----------
.. glossary::

    noise signal
        Formally, a noise signal is a signal produced by a stochastic (random) process.
        The aperiodic signals that are simulated in NeuroDSP are noise signals.
    coloured noise
        The 'colour' of noise refers the the power spectrum of a noise signal.

        * white noise: a signal with a flat power spectrum, with equal power at all frequencies.
        * pink noise: a signal with a 1/f power spectrum.
        * brown noise: a signal with a 1/f^2 power spectrum. Also called red noise.
    random walk
        A random process that describes a path of a succession of random steps.

