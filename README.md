# Neuro Digital Signal Processing Toolbox

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Latest Version](https://img.shields.io/pypi/v/neurodsp.svg)](https://pypi.python.org/pypi/neurodsp/)
[![Build Status](https://travis-ci.org/neurodsp-tools/neurodsp.svg)](https://travis-ci.org/neurodsp-tools/neurodsp)
[![codecov](https://codecov.io/gh/neurodsp-tools/neurodsp/branch/master/graph/badge.svg)](https://codecov.io/gh/neurodsp-tools/neurodsp)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/neurodsp.svg)](https://pypi.python.org/pypi/neurodsp/)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/neurodsp-tools/neurodsp/master)

A package of tools to process, analyze, and simulate neural recordings as individual voltage time series, with specific focus on time and frequency domain analyses.

## Documentation

Documentation for the NeuroDSP module is available [here](https://neurodsp-tools.github.io/neurodsp/).

## Dependencies

NeuroDSP is written in Python, and requires Python >= 3.5 to run.

It has the following dependencies:
- numpy
- scipy
- matplotlib
- scikit-learn
- pandas
- pytest (optional)

We recommend using the [Anaconda](https://www.continuum.io/downloads) distribution to manage these requirements.

## Install

To install the latest release of neurodsp, you can install from pip:

`$ pip install neurodsp`

To get the development version (updates that are not yet published to pip), you can clone this repo.

`$ git clone https://github.com/neurodsp-tools/neurodsp`

To install this cloned copy of neurodsp, move into the directory you just cloned, and run:

`$ pip install .`

## Modules

NeuroDSP includes the following modules, each of which have dedicated [tutorials](https://neurodsp-tools.github.io/neurodsp/auto_tutorials/index.html).

- ```filt``` : Filter data with bandpass, highpass, lowpass, or notch filters
- ```burst``` : Detect bursting oscillations in neural signals
- ```laggedcoherence``` : Estimate rhythmicity using the lagged coherence measure
- ```spectral``` : Compute spectral domain features such as power spectra
- ```swm``` : Identify recurrent patterns in signals using sliding window matching
- ```timefrequency``` : Estimate instantaneous measures of oscillatory activity
- ```sim``` : Simulate periodic and aperiodic signal components
- ```plts``` : Plotting functions
