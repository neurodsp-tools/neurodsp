"""Test function to compute lagged coherence."""

from neurodsp.rhythm.lc import *

###################################################################################################
###################################################################################################

def test_lagged_coherence(tsig):

    lcs = lagged_coherence(tsig, (8, 12), 100)
    assert True

## PRIVATE FUNCTIONS

def test_lagged_coherence_1freq():
    pass

def test_nonoverlapping_chunks():
    pass
