""".  """

from neurodsp.spectral.wavelet import *

###################################################################################################
###################################################################################################

def test_morlet_transform(tsig):

    out = morlet_transform(tsig, [5, 10, 15], fs=500)
    assert True

def test_morlet_convolve(tsig):

    out = morlet_convolve(tsig, 10, fs=500)
    assert True
