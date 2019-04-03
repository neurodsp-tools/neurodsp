# Contributing Guidelines

Thank you for your interest in contributing to NeuroDSP! We welcome contributions to the
project that extend or improve code and/or documenation and/or tutorials in the
NeuroDSP project!

If you have an idea you would like to contribute to NeuroDSP, please first check that
it is in scope for the project, as discussed below.

If it seems related to the project, it is best to go and open an
[issue](https://github.com/neurodsp-tools/neurodsp/issues),
suggesting your idea.

From there, you can follow the procedures and conventions described below to
add your contribution to NeuroDSP!

Note that contributors to NeuroDSP are expected to follow the
[Code of Conduct](https://github.com/neurodsp-tools/neurodsp/blob/master/CODE_OF_CONDUCT.md).

## Scope

`NeuroDSP` is a lightweight library for neural time series analysis.

As a rule of thumb, functions and algorithms that operate on neural time series are considered in scope.

Code and utilities that focused on data management, and/or multi-channel analyses are currently
considered out of scope for the project.

## Procedures

NeuroDSP is hosted and developed on Github.

To make a contribution:

1. Make an issue on the NeuroDSP repository, stating your intention and getting feedback from maintainers
2. Make a fork of the NeuroDSP repository
3. Update the fork of the repository with any updates and/or additions to the project
4. Check that any additions to the project follow the conventions described below
5. Make a pull request from your fork to the NeuroDSP repository
6. Address any feedback and/or recommendations from reviewers, until the contribution is ready to be merged

## Conventions

1. Code & Style
    * All code should be written in Python 3.4+
    * Code should follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide

2. Documentation
    * Docstrings for public functions should be in
[Numpy docs](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt) format.
At minimum, there should be a sentence describing what the function does and a list of
parameters and returns.
    * Private functions should be indicated with a leading underscore, and should still include a
docstrings including at least a sentence describition what the function does.
    * If you add any new public functions, note this function in the doc/api.rst file,
so that this function gets included in the documentation site API listing.

3. Dependencies
    * Any dependencies outside of the standard Anaconda distribution should be avoided if possible.
    * If any more packages are needed, they should be added to the `requirements.txt` file.

4. API & Naming Conventions
    * Try to keep the API consistent across neurodp in naming and parameter ordering, for example:
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
