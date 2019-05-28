"""Utility functions."""

from .data import create_times
from .norm import demean, normalize_variance
from .outliers import remove_nans, restore_nans, discard_outliers
