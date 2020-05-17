"""NeuroDSP setup script."""

import os
from setuptools import setup, find_packages

# Get the current version number from inside the module
with open(os.path.join('neurodsp', 'version.py')) as version_file:
    exec(version_file.read())

# Load the long description from the README
with open('README.rst') as readme_file:
    long_description = readme_file.read()

# Load the required dependencies from the requirements file
with open("requirements.txt") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name = 'neurodsp',
    version = __version__,
    description = 'Digital signal processing for neural time series.',
    long_description = long_description,
    python_requires = '>=3.5',
    author = 'The Voytek Lab',
    author_email = 'voyteklab@gmail.com',
    maintainer = 'Thomas Donoghue',
    maintainer_email = 'tdonoghue.research@gmail.com',
    url = 'https://github.com/neurodsp-tools/neurodsp',
    packages = find_packages(),
    license = 'Apache License, 2.0',
    download_url = 'https://github.com/neurodsp-tools/neurodsp/releases',
    keywords = ['neuroscience', 'neural oscillations', 'time series analysis', 'local field potentials',
                'spectral analysis', 'time frequency analysis', 'electrophysiology'],
    install_requires = install_requires,
    tests_require = ['pytest'],
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
        ],
    platforms = 'any',
    project_urls = {
        'Documentation' : 'https://neurodsp-tools.github.io/',
        'Bug Reports' : 'https://github.com/neurodsp-tools/neurodsp/issues',
        'Source' : 'https://github.com/neurodsp-tools/neurodsp'
    },
)
