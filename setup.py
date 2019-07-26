"""NeuroDSP setup script."""

import os
from setuptools import setup, find_packages

# Get the current version number from inside the module
with open(os.path.join('neurodsp', 'version.py')) as vf:
    exec(vf.read())

# Copy in long description.
#  Note: this is a partial copy from the README
#    Only update here in coordination with the README, to keep things consistent.
long_description = \
"""
========
Neurodsp
========

A package of tools to analyze and simulate neural time series, using digital signal processing.

Available modules in NeuroDSP include:

- filt : Filter data with bandpass, highpass, lowpass, or notch filters
- burst : Detect bursting oscillations in neural signals
- rhythm : Find and analyze rhythmic and recurrent patterns in time series
- spectral : Compute spectral domain features such as power spectra
- timefrequency : Estimate instantaneous measures of oscillatory activity
- sim : Simulate time series, including periodic and aperiodic signal components
- plts : Plotting functions
"""

setup(
    name = 'neurodsp',
    version = __version__,
    description = 'Digital Signal Processing for Neural time series',
    long_description = long_description,
    author = 'The Voytek Lab',
    author_email = 'voyteklab@gmail.com',
    url = 'https://github.com/neurodsp-tools/neurodsp',
    packages = find_packages(),
    license = 'MIT',
    download_url = 'https://github.com/neurodsp-tools/neurodsp/releases',
    keywords = ['neuroscience', 'neural oscillations', 'time series analysis', 'spectral analysis', 'LFP'],
    install_requires = ['numpy', 'scipy', 'matplotlib'],
    tests_require = ['pytest'],
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
        ]
)
