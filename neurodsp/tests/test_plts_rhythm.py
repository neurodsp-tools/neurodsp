"""Test rhythm plots."""

import numpy as np

from neurodsp.tests.utils import plot_test

from neurodsp.plts.rhythm import *

###################################################################################################
###################################################################################################

@plot_test
def tests_plot_swm_pattern():

    dat = np.arange(0, 2, 0.1)
    plot_swm_pattern(dat)

@plot_test
def test_plot_lagged_coherence():

    dat = np.arange(0, 2, 0.1)
    plot_lagged_coherence(dat, dat)
