""".  """

from neurodsp.filt.iir import *

###################################################################################################
###################################################################################################

def test_filter_signal_iir(tsig):

    sig = filter_signal_iir(tsig, 500, 'bandpass', (8, 12), 3)
    assert True

def test_design_iir_filter():
    pass
