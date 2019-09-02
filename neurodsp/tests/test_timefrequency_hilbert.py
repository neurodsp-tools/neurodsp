"""Test functions for timefrequency hilbert analyses."""

from neurodsp.tests.settings import FS

from neurodsp.timefrequency.hilbert import *

###################################################################################################
###################################################################################################

def test_robust_hilbert():

    # Generate a signal with NaNs
    fs, n_points, n_nans = 100, 1000, 10
    sig = np.random.randn(n_points)
    sig[0:n_nans] = np.nan

    # Check has correct number of nans (not all nan), without increase_n
    hilb_sig = robust_hilbert(sig)
    assert sum(np.isnan(hilb_sig)) == n_nans

    # Check has correct number of nans (not all nan), with increase_n
    hilb_sig = robust_hilbert(sig, True)
    assert sum(np.isnan(hilb_sig)) == n_nans

def test_phase_by_time(tsig):

    out = phase_by_time(tsig, FS, (8, 12))
    assert out.shape == tsig.shape

def test_amp_by_time(tsig):

    out = amp_by_time(tsig, FS, (8, 12))
    assert out.shape == tsig.shape

def test_freq_by_time(tsig):

    out = freq_by_time(tsig, FS, (8, 12))
    assert out.shape == tsig.shape

def test_no_filters(tsig):

    out = phase_by_time(tsig, FS)
    assert out.shape == tsig.shape

    out = amp_by_time(tsig, FS)
    assert out.shape == tsig.shape

    out = freq_by_time(tsig, FS)
    assert out.shape == tsig.shape

def test_2d(tsig2d):

    out = phase_by_time(tsig2d, FS, (8, 12))
    assert out.shape == tsig2d.shape

    out = amp_by_time(tsig2d, FS, (8, 12))
    assert out.shape == tsig2d.shape

    out = freq_by_time(tsig2d, FS, (8, 12))
    assert out.shape == tsig2d.shape
