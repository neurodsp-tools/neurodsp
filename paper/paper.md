---
title: 'NeuroDSP: A package for neural digital signal processing'
tags:
  - Python
  - neuroscience
  - digital signal processing
  - DSP
  - filtering
authors:
  - name: Scott Cole
    orcid: 0000-0002-6168-9951
    affiliation: 1,2
  - name: Thomas Donoghue
    orcid: 0000-0001-5911-0472
    affiliation: 1
  - name: Richard Gao
    orcid: 0000-0001-5916-6433
    affiliation: 1
  - name: Bradley Voytek
    orcid: 0000-0003-1640-2525
    affiliation: 1-4
affiliations:
 - name: Department of Cognitive Science, UC San Diego
   index: 1
 - name: Neurosciences Graduate Program, UC San Diego
   index: 2
 - name: Halıcıoğlu Data Science Institute, UC San Diego
   index: 3
 - name: Kavli Institute for Brain and Mind, UC San Diego
   index: 4   
date: 25 January 2019
bibliography: paper.bib
---

# Summary

Populations of neurons exhibit time-varying fluctuations in their aggregate activity. These data are often collected using common magneto- and electrophysiological methods, such as magneto or electroencephalography (M/EEG), intracranial EEG (iEEG) or electrocorticography (ECoG), and local field potential (LFP) recordings [REF]. While many Python tools exist for digital signal processing (DSP) [REF], neural data exhibit specific properties that warrant specialized analysis tools focused on idiosyncrasies of neural data. These include band-limited oscillations below 300-500 Hz [REF] and non-stationary bursting oscillations [REF], as well as an aperiodic signal that is variously referred to as the 1/f-like background, or noise, or scale-free activity [REF].

``NeuroDSP`` is a package of modules specifically designed to be used by neuroscientists analyzing neural time series data. The modules include:

* burst : Detect bursting oscillators in neural signals.
* filt : Filter data with bandpass, highpass, lowpass, or notch filters.
* laggedcoherence : Estimate rhythmicity using the lagged coherence measure.
* sim : Simulate bursting or stationary oscillators with brown noise.
* spectral : Compute spectral domain features (PSD and 1/f slope, etc).
* swm : Identify recurrent patterns in a signal using sliding window matching.
* timefrequency : Estimate instantaneous measures of oscillatory activity.

``NeuroDSP`` also includes detailed tutorials that are spread out across a series of notebooks, each of which demonstrates the sequence of analyses one might adopt using each of the modules. The tutorials include:

* Filtering: A general tutorial on filtering, filter parameters, and the ``NeuroDSP`` specific implementation.

* Instantaneous Amplitude, Frequency, and Phase: Computing instantaneous features, such as amplitude, phase, and frequency, from narrowband-filtered (putative) oscillations.

* Lagged Coherence: The ``NeuroDSP`` implementation of the lagged coherence algorithm for quantifying the presence of rhythms [@fransen_identifying_2015].

* Spectral Analysis: Computing and visualizing power spectral density with various methods, Morlet Wavelet Transform, and spectral coefficient of variation (SCV). For parametrizing the resulting spectrum, please see the companion spectral parametrization toolbox, fitting oscillations and one-over-f (FOOOF) [@haller_parameterizing_2018].

* Burst Detection: Burst detection using the dual threshold algorithm. For a more extensive and time-domain toolbox for detecting contiguous rhythmic cycles, please the companion toolbox, bycycle [@cole_cycle-by-cycle_2018, @cole_hippocampal_2018].

* Sliding Window Matching: The ``NeuroDSP`` implementation of the sliding window matching (SWM) algorithm for identifying recurring patterns in a neural signal, like the shape of an oscillatory waveform [@gips_discovering_2017].

* Simulating Oscillations and Noise: Simulating 1/f-like signals with various stochastic models, as well as stationary/bursty and non-sinusoidal oscillations.

#### Mentions (if applicable) of any ongoing research projects using the software or recent scholarly publications enabled by it



# Acknowledgements

Cole is supported by the National Science Foundation Graduate Research
Fellowship Program and the University of California, San Diego Chancellor’s
Research Excellence Scholarship. Gao is supported by the Natural Sciences and
Engineering Research Council of Canada (NSERC PGS-D), UCSD Kavli Innovative
Research Grant (IRG), and the Katzin Prize. Voytek is supported by a Sloan
Research Fellowship (FG-2015-66057), the Whitehall Foundation (2017-12-73),
and the National Science Foundation under grant BCS-1736028.
The authors declare no competing financial interests.

# References
