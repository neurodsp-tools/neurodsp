========================================
 Neuro Digital Signal Processing Toolbox
========================================

|ProjectStatus|_ |Version|_ |BuildStatus|_ |Coverage|_ |License|_ |PythonVersions|_ |Publication|_

.. |ProjectStatus| image:: https://www.repostatus.org/badges/latest/active.svg
.. _ProjectStatus: https://www.repostatus.org/#active

.. |Version| image:: https://img.shields.io/pypi/v/neurodsp.svg
.. _Version: https://pypi.python.org/pypi/neurodsp/

.. |BuildStatus| image:: https://travis-ci.com/neurodsp-tools/neurodsp.svg
.. _BuildStatus: https://travis-ci.com/github/neurodsp-tools/neurodsp

.. |Coverage| image:: https://codecov.io/gh/neurodsp-tools/neurodsp/branch/master/graph/badge.svg
.. _Coverage: https://codecov.io/gh/neurodsp-tools/neurodsp

.. |License| image:: https://img.shields.io/pypi/l/neurodsp.svg
.. _License: https://opensource.org/licenses/Apache-2.0

.. |PythonVersions| image:: https://img.shields.io/pypi/pyversions/neurodsp.svg
.. _PythonVersions: https://pypi.python.org/pypi/neurodsp/

.. |Publication| image:: https://joss.theoj.org/papers/10.21105/joss.01272/status.svg
.. _Publication: https://doi.org/10.21105/joss.01272

Tools to analyze and simulate neural time series, using digital signal processing.

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

``NeuroDSP`` is written in Python, and requires Python >= 3.5 to run.

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

Please use the
`Github issue tracker <https://github.com/neurodsp-tools/neurodsp/issues>`_
to file bug reports and/or ask questions about this project.

Modules
-------

Available modules in ``NeuroDSP`` include:

- ``filt`` : Filter data with bandpass, highpass, lowpass, or notch filters
- ``burst`` : Detect bursting oscillations in neural signals
- ``rhythm`` : Find and analyze rhythmic and recurrent patterns in time series
- ``spectral`` : Compute spectral domain features such as power spectra
- ``timefrequency`` : Estimate instantaneous measures of oscillatory activity
- ``sim`` : Simulate time series, including periodic and aperiodic signal components
- ``plts`` : Plotting functions

Contribute
----------

``NeuroDSP`` welcomes and encourages contributions from the community!

If you have an idea of something to add to NeuroDSP, please start by opening an
`issue <https://github.com/neurodsp-tools/neurodsp/issues>`_.

When writing code to add to NeuroDSP, please follow the
`Contribution Guidelines <https://github.com/neurodsp-tools/neurodsp/blob/master/CONTRIBUTING.md>`_.

We also require that all contributors follow our
`Code of Conduct <https://github.com/neurodsp-tools/neurodsp/blob/master/CODE_OF_CONDUCT.md>`_.

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

Supported by NIH award R01 GM134363

`NIGMS <https://www.nigms.nih.gov/>`_

.. image:: https://www.nih.gov/sites/all/themes/nih/images/nih-logo-color.png
  :width: 400
