NeuroDSP Glossary
=================

The following is a glossary of neuroscience and digital processing related terms that are used
in NeuroDSP, as well as links to some external resources for learning DSP.

General
-------

.. glossary::

    periodic
        Properties or components of a signal that are rhythmic.

    aperiodic
        Properties or components of a signal that are arrhythmic, with no characteristic frequency.

Digital Signal Processing
-------------------------

Digital Signal Processing (DSP) is an area of science & engineering related to the computational
analysis of digital signals.

For a collection of openly available resources for learning DSP, check out this
`list <https://github.com/openlists/DSPResources/>`_.

If you want to start with a general introduction to DSP, check out
`Seeing Circles Sines and Signal <https://jackschaedler.github.io/circles-sines-signals/>`_
or for more in-depth descriptions, check out
`The Scientist and Engineers Guide to Digital Signal Processing <http://www.dspguide.com/>`_.

.. glossary::

    time domain
        Signals that are represented as variations over time, and analyses of such signals.

    frequency domain
        Signals that are represented in terms of frequencies, and analyses of such signals.

    sampling rate
        The rate at which samples are taken.

    temporal resolution
        The precision of a measurement, in the time domain.
        This is set by the magnitude of time between successive measurements (e.g. 0.01 seconds between samples).

    frequency resolution
        The precision of a measurement, in the frequency domain.
        This is set by the magnitude of frequency between successive measurements (e.g. 0.5 Hz between measurements).

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

There are some available (pay-walled) articles that present overviews and guides to filters, including:

- this `guide <https://doi.org/10.1016/j.jneumeth.2014.08.002>`_ on using filters for electrophysiological data
- this `primer <https://doi.org/10.1016/j.neuron.2019.02.039>`_ on when, how, and why to use filters

For an open, in depth, and code-driven tutorial, check out the
`MNE Filtering Tutorial <https://mne.tools/stable/auto_tutorials/discussions/plot_background_filtering.html>`_.

.. glossary::

    Impulse Response
        The response of a filter when presented with an impulse; a single, brief input.

    FIR
        A Finite Impulse Response filter, meaning its impulse response settles to zero in finite time.

    IIR
        An Infinite Impulse Response filter, meaning the filter is recursive, and its impulse response continues infinitely.

    passband
        The range (band) of frequencies that are unattenuated by a filter.

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
        Periodic activity that lasts for a short or transient amount of time, as in a 'burst of oscillatory activity'.

Time Frequency
--------------

We currently have two general approaches to time frequency analyses:

* those based on the Hilbert transform

  * There is a scholarpedia article on using the
    `Hilbert Transform for Brain Waves <http://www.scholarpedia.org/article/Hilbert_transform_for_brain_waves>`_
  * See also this
    `deep dive into Hilbert methods <http://www.rdgao.com/roemerhasit_Hilbert_Transform/>`_
    from VoytekLab member Richard Gao.
* wavelet based approaches

.. glossary::

    frequency
        The number of occurrences over a unit of time, typically referred to as cycles per second, and measured in Hz.

    phase
        The position, at a point in time, on a waveform cycle.

    amplitude
        The magnitude of a signal, as the peak-to-trough distance.

    power
        The squared magnitude of a signal.

    period
        A single cycle of a rhythm, defined as the time between two consecutive troughs (or peaks).

    hilbert transform
        A mathematical transform that computes the 'analytic signal', a complex-valued representation
        of a time-series (signal) that can be used to find its analytic amplitude and phase.

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

    Fourier transform
        A mathematical transformation to decompose a time series into a frequency representation.

    power spectrum
        A frequency domain representation, as an estimate of the power across frequencies in a signal.

    median filter
        A smoothing approach to replace each value in a signal with the median of the neighboring entries.

    coefficient of variation
        A standardized measure of dispersion, as the ratio of the standard deviation to the mean.

Simulations
-----------

For an overview of the aperiodic signals available in terms of their 1/f characteristics, check out this
`article <http://www.scholarpedia.org/article/1/f_noise>`_
from scholarpedia.

.. glossary::

    noise signal
        Formally, a noise signal is a signal produced by a stochastic (random) process.
        The aperiodic signals that are simulated in NeuroDSP are, technically, noise signals.

    powerlaw
        A relationship between two quantities, whereby one quantity varies as a power of another.
        One-over-f relationships are powerlaw, as the spectral power varies by a power of the frequency.

    1/f signal
        A signal for which the power spectrum can be described by a powerlaw of the form :math:`1/f^\chi`,
        where :math:`\chi` refers to the exponent of the powerlaw.

    colored noise
        The 'color' of noise refers to the 1/f exponent of the power spectrum of a noise signal.

        * white noise: a signal with a :math:`1/f^0` power spectrum, which is flat with equal power across all frequencies
        * pink noise: a signal with a :math:`1/f^1` power spectrum
        * brown noise: a signal with a :math:`1/f^2` power spectrum, sometimes also known as red noise

    random walk
        A random process that describes a path of a succession of random steps.
