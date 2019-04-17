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
    affiliation: "1"
  - name: Thomas Donoghue
    orcid: 0000-0001-5911-0472
    affiliation: "2"
  - name: Richard Gao
    orcid: 0000-0001-5916-6433
    affiliation: "2"
  - name: Bradley Voytek
    orcid: 0000-0003-1640-2525
    affiliation: "1, 2, 3, 4"
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

Populations of neurons exhibit time-varying fluctuations in their aggregate activity. These data are often collected using common magneto- and electrophysiological methods, such as magneto or electroencephalography (M/EEG), intracranial EEG (iEEG) or electrocorticography (ECoG), and local field potential (LFP) recordings [@buzsaki_origin_2012]. While there are existing Python tools for digital signal processing (DSP), such as [scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html), neural data exhibit specific properties that warrant specialized analysis tools focused on idiosyncrasies of neural data. Features of interest in neural data include periodic properties—such as band-limited oscillations [@buzsaki_neural_2004] and transient or 'bursty' events—as well as an aperiodic signal that is variously referred to as the 1/f-like background [@freeman_simulated_2009; @miller_power-law_2009], or noise [@voytek_age-related_2015], or scale-free activity [@he_scale-free_2014], and that may carry information about the current generators, such as the ratio of excitation and inhibition [@gao_inferring_2017]. ``NeuroDSP`` is a package specifically designed to be used by neuroscientists for analyzing neural time series data, in particular for examing their time-varying properties related to oscillatory and 1/f-like components.

``NeuroDSP`` is accompanied by a [documentation site](https://neurodsp-tools.github.io/neurodsp/) that includes detailed [tutorials](https://neurodsp-tools.github.io/neurodsp/auto_tutorials/index.html#) for each of the modules, which are described below, as well as suggested workflows for combining them.

Modules in ``NeuroDSP`` include:

* filt : Filter data with bandpass, highpass, lowpass, or bandstop filters, using FIR or IIR filters.
* burst : Detect bursting oscillations in neural signals, for example using the dual threshold algorithm [@feingold_bursts_2015]. For a more extensive time-domain toolbox for detecting contiguous rhythmic cycles and calculating cycle-by-cycle features, please the companion toolbox, [ByCycle](https://bycycle-tools.github.io/bycycle/) [@cole_cycle-by-cycle_2018; @cole_hippocampal_2018].
* rhythm : Detect rhythmic patterns in neural time series. Algorithms to do so include the lagged coherence measure for quantifying the presence of rhythms [@fransen_identifying_2015], and the sliding window matching (SWM) algorithm for identifying recurring patterns in a neural signal, like the shape of an oscillatory waveform [@gips_discovering_2017].
* spectral : Compute spectral domain features, including power spectral estimation, mortlet wavelet transforms and spectral coefficient of variation (SCV). For parametrizing the resulting spectrum, please see the companion spectral parametrization toolbox, fitting oscillations and one-over-f or [FOOOF](https://fooof-tools.github.io/fooof/) [@haller_parameterizing_2018].
* timefrequency : Estimate instantaneous measures of oscillatory activity, including instantaneous measures for calculating the amplitude, frequency, and phase from narrowband-filtered, putative oscillations.
* sim : Simulate neural time series, including oscillations, that can vary in their waveform shape and stationarity, aperiodic signals, simulated with various stochastic models, transient events, as well as utilities to combine across various components.
* plts : Plotting functions.

# Statement of Need

``NeuroDSP`` is an open-source Python package for time-series analyses of neural data, including implementations of relevant DSP utilities as well as implementations of a collection of algorithms and approaches that have been developed specifically for neural data analysis. By design, ``NeuroDSP`` offers a lightweight architecture in which functions take in time-series directly, thus offering a flexible toolbox for custom analysis of a broad range of neural data. This approach complements, and can be used in conjunction with, related toolboxes such as MNE [@gramfort_mne_2014] that are more focused on data management and multi-channel analyses. ``NeuroDSP`` offers implementations of a distinct set of methods, with different use cases from other tools, and can easily be integrated with frameworks such as MNE or with other custom workflows. ``NeuroDSP`` also offers a developed module for simulating realistic field potential data, which is used for testing the properties of methods against synthetic data for which ground truth parameters are known. Note that these simulations are designed to mimic the statistics of electrophysiological data—and properties of transient, non-stationary, non-sinusoidal rhythms—but they should not be over-interpreted as biophysically realistic.

# Acknowledgements

We thank Andrew J. Washington for his early code contributions to this project. Cole is supported by the National Science Foundation Graduate Research Fellowship Program and the University of California, San Diego Chancellor’s Research Excellence Scholarship. Gao is supported by the Natural Sciences and Engineering Research Council of Canada (NSERC PGS-D), UCSD Kavli Innovative Research Grant (IRG), and the Katzin Prize. Voytek is supported by a Sloan Research Fellowship (FG-2015-66057), the Whitehall Foundation (2017-12-73), and the National Science Foundation under grant BCS-1736028. The authors declare no competing financial interests.

# References
