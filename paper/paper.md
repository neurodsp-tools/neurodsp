---
title: 'NeuroDSP: A package for neural digital signal processing'
tags:
  - Python
  - neuroscience
  - digital signal processing
  - DSP
  - filtering
  - time series analysis
  - electrophysiology
  - local field potentials
  - electroencephalography
  - EEG
  - magnetoencephalography
  - MEG
authors:
  - name: Scott Cole
    orcid: 0000-0002-6168-9951
    affiliation: 1
  - name: Thomas Donoghue
    orcid: 0000-0001-5911-0472
    affiliation: 2
  - name: Richard Gao
    orcid: 0000-0001-5916-6433
    affiliation: 2
  - name: Bradley Voytek
    orcid: 0000-0003-1640-2525
    affiliation: 1-4
affiliations:
 - name: Neurosciences Graduate Program, UC San Diego
   index: 1
 - name: Department of Cognitive Science, UC San Diego
   index: 2
 - name: Halıcıoğlu Data Science Institute, UC San Diego
   index: 3
 - name: Kavli Institute for Brain and Mind, UC San Diego
   index: 4
date: 08 February 2019
bibliography: paper.bib
---

# Summary

Populations of neurons exhibit time-varying fluctuations in their aggregate electrical electrophysiological methods, such as magneto or electroencephalography (M/EEG), intracranial EEG (iEEG) or electrocorticography (ECoG), and local field potential (LFP) recordings [@buzsaki_origin_2012]. While there are existing Python tools for digital signal processing (DSP), such as [scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html), neural data exhibit specific properties that warrant specialized analysis tools focused on idiosyncrasies of neural data. Features of interest in neural data include periodic properties—-such as band-limited oscillations [@buzsaki_neural_2004] and transient or 'bursty' events--as well as an aperiodic signal that is variously referred to as the 1/f-like background [@freeman_simulated_2009, @miller_power-law_2009], or noise [@voytek_age-related_2015], or scale-free activity [@he_scale-free_2014], and that may carry information about the current generators, such as the ratio of excitation and inhibition [@gao_interpreting_2016].

``NeuroDSP`` is an open-source Python package for time-series analyses of neural data, including implementations of relevant DSP utilities as well as implementations of a collection of algorithms and approaches that have been developed specifically for neural data analysis. ``NeuroDSP`` complements, and can be used in conjunction with, related toolboxes such as MNE [@gramfort_mne_2014], as it offers implementations of a distinct set of methods. By design, ``NeuroDSP`` offers a lightweight design in which functions take in time-series directly so can be easily integrated with tools such as MNE and with custom workflows. ``NeuroDSP`` also offers a developed module for simulating realistic neural data, which be used for testing the properties of methods against synthetic data for which ground truth parameters are known.

# Features

``NeuroDSP`` is a package specifically designed to be used by neuroscientists analyzing neural time series data. The modules include:

* burst : Detect bursting oscillators in neural signals.
* filt : Filter data with bandpass, highpass, lowpass, or notch filters.
* laggedcoherence : Estimate rhythmicity using the lagged coherence measure.
* sim : Simulate bursting or stationary oscillators along with aperiodic signals.
* spectral : Compute spectral domain features (power spectra and aperiodic slope, etc).
* swm : Identify recurrent patterns in a signal using sliding window matching.
* timefrequency : Estimate instantaneous measures of oscillatory activity.

# Documentation

``NeuroDSP`` is accompanied by a [documentation site](https://neurodsp-tools.github.io/neurodsp/) that includes detailed [tutorials](https://neurodsp-tools.github.io/neurodsp/auto_tutorials/index.html#) that demonstrate how the use the included analysis approaches, and that also demonstrates the sequence of analyses one might adopt using each of the modules.

The tutorials include:

* Filtering : A general tutorial on filtering, filter parameters, and the ``NeuroDSP`` specific implementation.

* Instantaneous Amplitude, Frequency, and Phase : Computing instantaneous features, such as amplitude, phase, and frequency, from narrowband-filtered, putative oscillations.

* Lagged Coherence : The ``NeuroDSP`` implementation of the lagged coherence algorithm for quantifying the presence of rhythms [@fransen_identifying_2015].

* Spectral Analysis : Computing and visualizing power spectral density with various methods, Morlet Wavelet Transform, and spectral coefficient of variation (SCV). For parametrizing the resulting spectrum, please see the companion spectral parametrization toolbox, fitting oscillations and one-over-f (FOOOF) [@haller_parameterizing_2018].

* Burst Detection : Burst detection using the dual threshold algorithm. For a more extensive time-domain toolbox for detecting contiguous rhythmic cycles and calculating cycle-by-cycle features, please the companion toolbox, bycycle [@cole_cycle-by-cycle_2018, @cole_hippocampal_2018].

* Sliding Window Matching : The ``NeuroDSP`` implementation of the sliding window matching (SWM) algorithm for identifying recurring patterns in a neural signal, like the shape of an oscillatory waveform [@gips_discovering_2017].

* Simulating Oscillations and Noise : Simulating aperiodic (1/f-like) signals with various stochastic models, as well as oscillations that can vary in their waveform shape and stationarity.

# Acknowledgements

Cole is supported by the National Science Foundation Graduate Research Fellowship Program and the University of California, San Diego Chancellor’s Research Excellence Scholarship. Gao is supported by the Natural Sciences and Engineering Research Council of Canada (NSERC PGS-D), UCSD Kavli Innovative Research Grant (IRG), and the Katzin Prize. Voytek is supported by a Sloan Research Fellowship (FG-2015-66057), the Whitehall Foundation (2017-12-73), and the National Science Foundation under grant BCS-1736028. The authors declare no competing financial interests.

# References
