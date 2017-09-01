# Contribution guidelines

This file is an incomplete outline of the expectations when contributing to `neurodsp`.

1. Documentation
    * Docstrings of outward-facing functions should be in
[Numpy docs](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt) format. 
At minimum, there should be a sentence describing what the function does and a list of
parameters and returns.
    * Docstrings of non-outward-facing functions should begin with an underscore
and should have at least a sentence describition what the function does.

2. PEP8
    * Code should follow the pep8 style guide
    * All feedback from pep8speaks on the pull request must be satisfied

3. Tests
    * At a minimum, consistency tests should be written 
    * Thorough testing of functionality is welcome, but not a requirement at this time.
    * Tests are run on python 3.4 and above using Travis CI.
    * TravisCI must pass all tests before pull request is merged.
    
4. Tutorials
    * For each module, a Jupyter notebook should illustrate the use of all functions within that module using an example neural signal
    
5. Dependencies
    * Any dependencies outside of the standard Anaconda distribution should be avoided if possible.
    * If any more packages are needed, they should be added to the `requirements.txt` file.
    
6. Standard API
    * Try to keep the API consistent with other modules, for example:
        * `x` is commonly the first input, representing the neural time series
        * `Fs` is commonly the second input, representing the sampling rate

7. Naming conventions
    * Try to keep naming conventions consistent with other modules, for example:
        * Function names are in all lowercase with underscores
        * Variables named 'f_...' refer to frequencies (e.g. 'f_lo' is a bandpass filter's low-frequency cutoff)
