"""Default settings for tests."""

import os
from pathlib import Path

import numpy as np

###################################################################################################
###################################################################################################

# Define general settings for test simulations
FS = 100
N_SECONDS = 10.0

FS_ODD = 123
N_SECONDS_ODD = 1/7

N_SECONDS_CYCLE = 1.0

# Define parameter options for test simulations
FREQ1 = 10
FREQ2 = 25
FREQ_SINE = 1
FREQS_LST = [8, 12, 1]
FREQS_ARR = np.array([5, 10, 15])
EXP1 = -1
EXP2 = -2
KNEE = 100

# Define settings for testing analyses
F_RANGE = (FREQ1-2, FREQ1+2)

# Define error tolerance levels for test comparisons
EPS = 10**(-10)
EPS_FILT = 10**(-3)

# Set paths for test files
TESTS_PATH = Path(os.path.abspath(os.path.dirname(__file__)))
BASE_TEST_FILE_PATH = TESTS_PATH / 'test_files'
TEST_PLOTS_PATH = BASE_TEST_FILE_PATH / 'plots'
TEST_FILES_PATH = BASE_TEST_FILE_PATH / 'files'
