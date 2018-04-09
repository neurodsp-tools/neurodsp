"""neurodsp setup script"""
from setuptools import setup, find_packages
setup(
    name = 'neurodsp',
    version = '0.3',
    description = 'A package of modules for analyzing neural signals',
    author = 'The Voytek Lab',
    author_email = 'voyteklab@gmail.com',
    url = 'https://github.com/voytekresearch/neurodsp',
    packages=find_packages(),
    license='MIT',
    download_url = 'https://github.com/voytekresearch/neurodsp/archive/0.3.tar.gz',
    keywords = ['neuroscience', 'neural oscillations', 'time series analysis', 'spectral analysis', 'LFP'],
    classifiers = []
)
