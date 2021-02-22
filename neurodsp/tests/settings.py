"""Default settings for tests."""

import os
import pkg_resources as pkg

import numpy as np

###################################################################################################
###################################################################################################

# Define general settings for test simulations
FS = 100
FS_HIGH = 1000
N_SECONDS = 1.0
N_SECONDS_LONG = 10.0

# Define parameter options for test simulations
FREQ1 = 10
FREQ2 = 25
FREQ_SINE = 1
FREQS_LST = [8, 12, 1]
FREQS_ARR = np.array([5, 10, 15])
EXP1 = -1
EXP2 = -2
KNEE = 10**2

# Define error tolerance levels for test comparisons
EPS = 10**(-10)
EPS_FILT = 10**(-3)

# Set paths for test files
BASE_TEST_FILE_PATH = pkg.resource_filename(__name__, 'test_files')
TEST_PLOTS_PATH = os.path.join(BASE_TEST_FILE_PATH, 'plots')
