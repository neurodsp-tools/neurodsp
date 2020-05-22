# Contributing Guidelines

Thank you for your interest in contributing to NeuroDSP! 

We welcome contributions to the project that extend or improve code and/or documentation 
in the module!

Note that contributors to NeuroDSP are expected to follow the
[Code of Conduct](https://github.com/neurodsp-tools/neurodsp/blob/master/CODE_OF_CONDUCT.md).

## How to Contribute

If you are interested in getting involved in helping with and maintaining `NeuroDSP`, the best place to start is on the [issues](https://github.com/neurodsp-tools/neurodsp/issues) board. Check through if there are any open issues that you might be able to contribute to!

If you have a new idea you would like to contribute to NeuroDSP, please do the following:

1. Check that your idea is in scope for the project (as discussed below).
2. Open an [issue](https://github.com/neurodsp-tools/neurodsp/issues) in the repository suggesting and describing your idea
3. Wait for some feedback from project maintainers on the idea, and coordinate with maintainers an a plan for the contribution
   
To contribute a fix, or add an update of the repository, do the following:

1. Make a fork of the NeuroDSP repository
2. Update the fork of the repository with any updates and/or additions to the project
3. Check that any additions to the project follow the conventions (described below)
4. Make a pull request from your fork to the `NeuroDSP` repository
5. Address any feedback and/or recommendations from reviewers, until the contribution is ready to be merged

## Scope

All contributions to `NeuroDSP` must be within the scope of the module. 

The scope of `NeuroDSP` is digital signal processing approaches for neural data, that operate on neural time series. 

The goal of `NeuroDSP` is to provide a light-weight collection of useful analyses approaches, agnostic to precise data types, and data organizations. Code and utilities that focused on data management are considered out of scope. 

## Conventions

All code contributed to the module should follow these conventions:

1. Code & Style
    * All code should be written in Python 3.5+
    * Code should follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide

2. Documentation
    * Docstrings for public functions should be in
[Numpy docs](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt) format.
At minimum, there should be a sentence describing what the function does and a list of
parameters and returns.
    * Private functions should be indicated with a leading underscore, and should still include a
docstrings including at least a sentence describing what the function does.
    * If you add any new public functions, note this function in the doc/api.rst file,
so that this function gets included in the documentation site API listing.

3. Dependencies
    * Any dependencies outside of the standard Anaconda distribution should be avoided if possible.
    * If any more packages are needed, they should be added to the `requirements.txt` file.

4. API & Naming Conventions
    * Try to keep the API consistent across NeuroDSP in naming and parameter ordering, for example:
        * `sig` is commonly the first input, representing the neural time series
        * `fs` is commonly the second input, representing the sampling rate
    * Try to keep naming conventions consistent with other modules, for example:
        * Function names are in all lowercase with underscores
        * Variables named 'f_...' refer to frequencies (e.g. 'f_range' is a bandpass filter's cutoff frequencies)

5. Tests
    * All code within NeuroDSP requires test code that executes that code
    * These tests, at a minimum, must be 'smoke tests' that execute the
code and check that it runs through, without erroring out, and returning appropriate variables.
    * If possible, including more explicit test code that checks more stringently for accuracy is encouraged,
but not strictly required.
    * Before a pull request is merged, code coverage must demonstrate that new code is test,
and continuous integration checks running this test code must all pass

6. Tutorials
    * If a new function or module is added, a quick tutorial demonstration of using this
code should be added to the tutorials section.
