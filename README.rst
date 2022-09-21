========================================
 Neuro Digital Signal Processing Toolbox
========================================

|ProjectStatus|_ |Version|_ |BuildStatus|_ |Coverage|_ |License|_ |PythonVersions|_ |Publication|_

.. |ProjectStatus| image:: https://www.repostatus.org/badges/latest/active.svg
.. _ProjectStatus: https://www.repostatus.org/#active

.. |Version| image:: https://img.shields.io/pypi/v/neurodsp.svg
.. _Version: https://pypi.python.org/pypi/neurodsp/

.. |BuildStatus| image:: https://github.com/neurodsp-tools/neurodsp/actions/workflows/build.yml/badge.svg
.. _BuildStatus: https://github.com/neurodsp-tools/neurodsp/actions/workflows/build.yml

.. |Coverage| image:: https://codecov.io/gh/neurodsp-tools/neurodsp/branch/main/graph/badge.svg
.. _Coverage: https://codecov.io/gh/neurodsp-tools/neurodsp

.. |License| image:: https://img.shields.io/pypi/l/neurodsp.svg
.. _License: https://opensource.org/licenses/Apache-2.0

.. |PythonVersions| image:: https://img.shields.io/pypi/pyversions/neurodsp.svg
.. _PythonVersions: https://pypi.python.org/pypi/neurodsp/

.. |Publication| image:: https://joss.theoj.org/papers/10.21105/joss.01272/status.svg
.. _Publication: https://doi.org/10.21105/joss.01272

Tools to analyze and simulate neural time series, using digital signal processing.

Overview
--------

`neurodsp` is a collection of approaches for applying digital signal processing, and
related algorithms, to neural time series. It also includes simulation tools for generating
plausible simulations of neural time series.

Available modules in ``NeuroDSP`` include:

- ``filt`` : Filter data with bandpass, highpass, lowpass, or notch filters
- ``timefrequency`` : Estimate instantaneous measures of oscillatory activity
- ``spectral`` : Compute freqeuncy domain features such as power spectra
- ``burst`` : Detect bursting oscillations in neural signals
- ``rhythm`` : Find and analyze rhythmic and recurrent patterns in time series
- ``aperiodic`` : Analyze aperiodic features of neural time series
- ``sim`` : Simulate time series, including periodic and aperiodic signal components
- ``plts`` : Plot neural time series and derived measures
- ``utils`` : Additional utilities for managing time series data

Documentation
-------------

Documentation for the ``NeuroDSP`` module is available `here <https://neurodsp-tools.github.io/neurodsp/>`_.

The documentation includes:

- `Tutorials <https://neurodsp-tools.github.io/neurodsp/auto_tutorials/index.html>`_: which describe and work through each module in NeuroDSP
- `Examples <https://neurodsp-tools.github.io/neurodsp/auto_examples/index.html>`_: demonstrating example applications and workflows
- `API List <https://neurodsp-tools.github.io/neurodsp/api.html>`_: which lists and describes all the code and functionality available in the module
- `Glossary <https://neurodsp-tools.github.io/neurodsp/glossary.html>`_: which defines all the key terms used in the module

If you have a question about using NeuroDSP that doesn't seem to be covered by the documentation, feel free to
open an `issue <https://github.com/neurodsp-tools/neurodsp/issues>`_ and ask!

Dependencies
------------

``NeuroDSP`` is written in Python, and requires Python >= 3.6 to run.

It has the following dependencies:

- `numpy <https://github.com/numpy/numpy>`_
- `scipy <https://github.com/scipy/scipy>`_
- `matplotlib <https://github.com/matplotlib/matplotlib>`_

Optional dependencies:

- `pytest <https://github.com/pytest-dev/pytest>`_ is needed if you want to run the test suite locally

We recommend using the `Anaconda <https://www.anaconda.com/products/individual>`_ distribution to manage these requirements.

Install
-------

The current major release of NeuroDSP is the 2.X.X series.

See the `changelog <https://neurodsp-tools.github.io/neurodsp/changelog.html>`_ for notes on major version releases.

**Stable Release Version**

To install the latest stable release, you can use pip:

.. code-block:: shell

    $ pip install neurodsp

NeuroDSP can also be installed with conda, from the conda-forge channel:

.. code-block:: shell

    $ conda install -c conda-forge neurodsp

**Development Version**

To get the current development version, first clone this repository:

.. code-block:: shell

    $ git clone https://github.com/neurodsp-tools/neurodsp

To install this cloned copy, move into the directory you just cloned, and run:

.. code-block:: shell

    $ pip install .

**Editable Version**

To install an editable version, download the development version as above, and run:

.. code-block:: shell

    $ pip install -e .

Contribute
----------

This project welcomes and encourages contributions from the community!

To file bug reports and/or ask questions about this project, please use the
`Github issue tracker <https://github.com/neurodsp-tools/neurodsp/issues>`_.

To see and get involved in discussions about the module, check out:

- the `issues board <https://github.com/neurodsp-tools/neurodsp/issues>`_ for topics relating to code updates, bugs, and fixes
- the `development page <https://github.com/neurodsp-tools/Development>`_ for discussion of potential major updates to the module

When interacting with this project, please use the
`contribution guidelines <https://github.com/neurodsp-tools/fooof/blob/main/CONTRIBUTING.md>`_
and follow the
`code of conduct <https://github.com/neurodsp-tools/neurodsp/blob/main/CODE_OF_CONDUCT.md>`_.

Reference
---------

If you use this code in your project, please cite:

.. code-block:: text

    Cole, S., Donoghue, T., Gao, R., & Voytek, B. (2019). NeuroDSP: A package for
    neural digital signal processing. Journal of Open Source Software, 4(36), 1272.
    DOI: 10.21105/joss.01272

Direct Link: https://doi.org/10.21105/joss.01272

Bibtex:

.. code-block:: text

    @article{cole_neurodsp:_2019,
        title = {NeuroDSP: A package for neural digital signal processing},
        author = {Cole, Scott and Donoghue, Thomas and Gao, Richard and Voytek, Bradley},
        journal = {Journal of Open Source Software},
        year = {2019},
        volume = {4},
        number = {36},
        issn = {2475-9066},
        url = {https://joss.theoj.org/papers/10.21105/joss.01272},
        doi = {10.21105/joss.01272},
    }

Funding
-------

Supported by NIH award R01 GM134363 from the
`NIGMS <https://www.nigms.nih.gov/>`_.

.. image:: https://www.nih.gov/sites/all/themes/nih/images/nih-logo-color.png
  :width: 400

|
