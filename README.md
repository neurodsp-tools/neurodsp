# neurodsp

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Latest Version](https://img.shields.io/pypi/v/neurodsp.svg)](https://pypi.python.org/pypi/neurodsp/)
[![Build Status](https://travis-ci.org/voytekresearch/neurodsp.svg)](https://travis-ci.org/voytekresearch/neurodsp)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/neurodsp.svg)](https://pypi.python.org/pypi/neurodsp/)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/voytekresearch/neurodsp/master)

A package of tools to process, analyze, and simulate neural recordings as individual voltage time series, with specific focus on time and frequency domain analyses. The primary purpose of this library is to serve as the shared codebase of common analyses for the [Voytek Lab](http://voyteklab.com/), but we welcome anyone's use and contributions.

## Python version support
This package has been tested on python 3.5, and 3.6 with the latest [Anaconda](https://www.continuum.io/downloads) distribution. Support for python 2 and earlier versions of python 3 is not guaranteed.

## Install

To install the latest release of neurodsp, you can install from pip:

`$ pip install neurodsp`

To get the development version (updates that are not yet published to pip), you can clone this repo.

`$ git clone https://github.com/voytekresearch/neurodsp.git`

To install this cloned copy of neurodsp, move into the directory you just cloned, and run:

`$ pip install .`

## Modules

- ```burst``` : Detect bursting oscillators in neural signals ([Tutorial](https://github.com/voytekresearch/neurodsp/blob/master/tutorials/5-BurstDetection.ipynb))
- ```filt``` : Filter data with bandpass, highpass, lowpass, or notch filters ([Tutorial](https://github.com/voytekresearch/neurodsp/blob/master/tutorials/1-Filtering.ipynb))
- ```laggedcoherence``` : Estimate rhythmicity using the lagged coherence measure ([Tutorial](https://github.com/voytekresearch/neurodsp/blob/master/tutorials/3-LaggedCoherence.ipynb))
- ```sim``` : Simulate bursting or stationary oscillators with brown noise ([Tutorial](https://github.com/voytekresearch/neurodsp/blob/master/tutorials/7-SimulatingSignals.ipynb))
- ```spectral``` : Compute spectral domain features (PSD and 1/f slope, etc) ([Tutorial](https://github.com/voytekresearch/neurodsp/blob/master/tutorials/4-SpectralAnalysis.ipynb))
- ```swm``` : Identify recurrent patterns in a signal using sliding window matching ([Tutorial](https://github.com/voytekresearch/neurodsp/blob/master/tutorials/6-SlidingWindowMatching.ipynb))
- ```timefrequency``` : Estimate instantaneous measures of oscillatory activity ([Tutorial](https://github.com/voytekresearch/neurodsp/blob/master/tutorials/2-InstantaneousMeasures.ipynb))

## Dependencies

- numpy
- scipy
- matplotlib
- scikit-learn
- pytest (optional)
