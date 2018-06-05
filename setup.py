"""neurodsp setup script"""
from setuptools import setup, find_packages

# Copy in long description.
#  Note: this is a partial copy from the README
#    Only update here in coordination with the README, to keep things consistent.
long_description = \
"""
========
Neurodsp
========
A package of modules to process and analyze neural recordings as individual voltage time series.
The primary purpose of this library is to serve as the shared codebase for the Voytek lab,
but we welcome anyone's use and contributions.

Neurodsp contains several modules:
- filt : Filter data with bandpass, highpass, lowpass, or notch filters
- spectral : Compute spectral domain features (PSD and 1/f slope, etc)
- timefrequency : Estimate instantaneous measures of oscillatory activity
- shape : Measure the waveform shape of neural oscillations
	- cyclefeatures : Compute features of an oscillation on a cycle-by-cycle basis
	- cyclepoints : Identify the extrema and zerocrossings for each cycle
	- phase : Estimate instantaneous phase by interpolating between extrema and zerocrossings
	- swm : Identify recurrent patterns in a signal using sliding window matching
- burst : Detect bursting oscillators in neural signals
- sim : Simulate bursting or stationary oscillators with brown noise
- pac : Estimate phase-amplitude coupling between two frequency bands
- laggedcoherence : Estimate rhythmicity using the lagged coherence measure
"""

setup(
    name = 'neurodsp',
    version = '0.3.1',
    description = 'A package of modules for analyzing neural signals',
    author = 'The Voytek Lab',
    author_email = 'voyteklab@gmail.com',
    url = 'https://github.com/voytekresearch/neurodsp',
    packages=find_packages(),
    license='MIT',
    download_url = 'https://github.com/voytekresearch/neurodsp/archive/0.3.1.tar.gz',
    keywords = ['neuroscience', 'neural oscillations', 'time series analysis', 'spectral analysis', 'LFP'],
    classifiers = []
)
