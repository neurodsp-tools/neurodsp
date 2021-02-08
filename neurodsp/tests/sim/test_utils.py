"""Tests for neurodsp.sim.utils."""

from neurodsp.sim.utils import *

###################################################################################################
###################################################################################################

def test_check_osc_def():

	n_seconds = 1.0
	fs = 100
	freq = 10

	# Check definition that should pass
	assert check_osc_def(n_seconds, fs, freq)

	# Check a definition that should fail the sampling rate check
	assert not check_osc_def(1.05, fs, freq)

	# Check a definition that should fail the time check
	assert not check_osc_def(n_seconds, 1111, freq)

	# Check a definition that should fail both checks
	assert not check_osc_def(1.05, 1111, freq)
