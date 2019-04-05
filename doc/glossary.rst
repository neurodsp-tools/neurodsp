NeuroDSP Glossary
=================

The following is a glossary of neuroscience and digital processing related terms that are used in NeuroDSP.

General
-------

.. glossary::

    periodic
        Properties or components of a signal that are rhythmic.
    aperiodic
        Properties or components of a signal that are arrhythmic, with no characteristic frequency.

Digital Signal Processing
-------------------------

For a general introduction to digital signal processing, we recommend
`Seeing Circles Sines and Signal <https://jackschaedler.github.io/circles-sines-signals/>`_
by Jack Schaedler.

.. glossary::

    time domain
        Signals that are represented as variations over time, and analyses of such signals.
    frequency domain
        Signals that are represented in terms of frequencies, and Analyses of such signals.
    temporal resolution
        The precision of a measurement, in the time domain.
    frequency resolution
        The precision of a measurement, in the frequency domain.
    sampling rate
        The rate at which samples are taken.

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

For a guide on filtering, specific to electrophysiological data, check out this
`paper <https://doi.org/10.1016/j.jneumeth.2014.08.002>`_ from the journal of neuroscience methods.

For a more in depth tutorial, in code, check out the
`MNE Filtering Tutorial <https://martinos.org/mne/stable/auto_tutorials/plot_background_filtering.html>`_.

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

We currently have two general approaches to time frequency analyses:

* those based on the Hilbert transform

  * There is a scholarpedia article on using the
    `Hilbert Transform for Brain Waves <http://www.scholarpedia.org/article/Hilbert_transform_for_brain_waves>`_
  * See also this
    `deep dive into Hilbert methods <http://www.rdgao.com/roemerhasit_Hilbert_Transform/>`_
    from VoytekLab member Richard Gao.
* wavelet based approaches.

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

Many of the spectral methods available are based on the Fourier transform, for which there is an
`interactive guide <https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/>`_
by Better Explained and an
`explainer video <https://www.youtube.com/watch?v=spUNpyF58BY>`_
by 3Blue1Brown.

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

For an overview of the aperiodic signals avaible in terms of their 1/f characteristics, check out this
`article <http://www.scholarpedia.org/article/1/f_noise>`_
from scholarpedia.

.. glossary::

    noise signal
        Formally, a noise signal is a signal produced by a stochastic (random) process.
        The aperiodic signals that are simulated in NeuroDSP are noise signals.
    powerlaw
        A relationship between two quantities, whereby one quantity varies as a power of another.
        One-over-f relationships are powerlaw, as the spectral power varies by a power of the frequency.
    1/f signal
        A signal distributed signal is one in which the power spectrum of the signal can be described by a 1/f^chi powerlaw.
    coloured noise
        The 'colour' of noise refers the the power spectrum of a noise signal.

        * white noise: a signal with a flat power spectrum, with equal power at all frequencies.
        * pink noise: a signal with a 1/f power spectrum. Pink noise can be considered to be a powerlaw with exponent of 1.
        * brown noise: a signal with a 1/f^2 power spectrum. Also called red noise.
    random walk
        A random process that describes a path of a succession of random steps.
