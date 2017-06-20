# neurodsp

[![Build Status](https://travis-ci.org/srcole/neurodsp.svg)](https://travis-ci.org/srcole/neurodsp)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)

A package of modules to process and analyze neural recordings as individual voltage time series. The primary purpose of this library is to serve as the shared codebase for the [Voytek Lab](http://voyteklab.com/), but we welcome anyone's use and contributions.

## Python version support
This package currently supports python 3 (tested on 3.6), but not python 2.

## Get latest code

`$ git clone https://github.com/voytekresearch/neurodsp.git`

## Install latest release of neurodsp

`$ pip install neurodsp`

## Modules

- filt : Bandpass, highpass, lowpass, and notch filters ([Tutorial](https://github.com/voytekresearch/neurodsp/blob/master/tutorials/Filtering.ipynb))
- laggedcoherence : Estimation of rhythmicity using the lagged coherence measure ([Tutorial](https://github.com/voytekresearch/neurodsp/blob/master/tutorials/Lagged%20Coherence.ipynb))
- timefrequency : Estimate instantaneous measures of oscillatory activity ([Tutorial](https://github.com/voytekresearch/neurodsp/blob/master/tutorials/Instantaneous%20measures%20of%20phase%20amplitude%20and%20frequency.ipynb)) 
- shape : submodules for measuring the waveform shape of neural oscillations
	- cyclefeatures : Compute features of an oscillation on a cycle-by-cycle basis ([Tutorial](https://github.com/voytekresearch/neurodsp/blob/master/tutorials/Cycle-by-cycle%20features%20of%20oscillatory%20waveforms.ipynb)) 
	- cyclepoints : Identify the extrema and zerocrossings for each cycle ([Tutorial](https://github.com/voytekresearch/neurodsp/blob/master/tutorials/Extrema%20and%20zerocross%20finding.ipynb)) 
	- phase : Estimate instantaneous phase by interpolating between extrema and zerocrossings ([Tutorial](https://github.com/voytekresearch/neurodsp/blob/master/tutorials/Phase%20estimation%20by%20extrema%20interpolation.ipynb)) 
	- swm : Identify recurrent patterns in a signal using sliding window matching ([Tutorial](https://github.com/voytekresearch/neurodsp/blob/master/tutorials/Sliding%20Window%20Matching.ipynb)) 

## Dependencies

- numpy
- scipy
- matplotlib
- pandas
- pytest
