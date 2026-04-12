"""
Test suite for IRASA multi-method support fix.

This test file validates that compute_irasa now supports all spectral
estimation methods: welch, medfilt, multitaper, and wavelet.

Before patch: Only welch worked; others raised AssertionError or KeyError.
After patch: All methods work with appropriate parameters.
"""

import numpy as np
import pytest
from neurodsp.sim import sim_combined
from neurodsp.aperiodic import compute_irasa


class TestIRASAMultiMethods:
    """Test compute_irasa with all supported spectral methods."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create test signal and frequency range."""
        self.fs = 50.0
        self.sig = sim_combined(
            n_seconds=60,
            fs=self.fs,
            components={
                'sim_powerlaw': {},
                'sim_oscillation': {'freq': 10.0}
            }
        )
        self.f_range = [1, 20]

    def test_irasa_welch(self):
        """Test IRASA with Welch method (original working case)."""
        freqs, ap, pe = compute_irasa(
            self.sig, self.fs,
            f_range=self.f_range,
            method='welch',
            nperseg=1000,
            noverlap=500
        )
        assert len(freqs) > 0, "Welch returned empty frequency array"
        assert len(ap) == len(freqs), "Aperiodic component length mismatch"
        assert len(pe) == len(freqs), "Periodic component length mismatch"

    def test_irasa_medfilt(self):
        """Test IRASA with medfilt method (patched)."""
        freqs, ap, pe = compute_irasa(
            self.sig, self.fs,
            f_range=self.f_range,
            method='medfilt',
            filt_len=1.0
        )
        assert len(freqs) > 0, "Medfilt returned empty frequency array"
        assert len(ap) == len(freqs), "Aperiodic component length mismatch"
        assert len(pe) == len(freqs), "Periodic component length mismatch"
        assert np.isfinite(ap).all(), "Aperiodic component has non-finite values"
        assert np.isfinite(pe).all(), "Periodic component has non-finite values"

    def test_irasa_multitaper(self):
        """Test IRASA with multitaper method (patched)."""
        freqs, ap, pe = compute_irasa(
            self.sig, self.fs,
            f_range=self.f_range,
            method='multitaper'
        )
        assert len(freqs) > 0, "Multitaper returned empty frequency array"
        assert len(ap) == len(freqs), "Aperiodic component length mismatch"
        assert len(pe) == len(freqs), "Periodic component length mismatch"

    def test_irasa_wavelet(self):
        """Test IRASA with wavelet method (patched)."""
        freqs_wavelet = np.logspace(
            np.log10(self.f_range[0]),
            np.log10(self.f_range[1]),
            30
        )
        freqs, ap, pe = compute_irasa(
            self.sig, self.fs,
            f_range=self.f_range,
            method='wavelet',
            freqs=freqs_wavelet,
            n_cycles=3.0
        )
        assert len(freqs) > 0, "Wavelet returned empty frequency array"
        assert len(ap) == len(freqs), "Aperiodic component length mismatch"
        assert len(pe) == len(freqs), "Periodic component length mismatch"

    def test_irasa_default_method_is_welch(self):
        """Verify backward compatibility: default method is welch."""
        freqs_default, ap_default, pe_default = compute_irasa(
            self.sig, self.fs,
            f_range=self.f_range,
            nperseg=1000
        )
        freqs_explicit, ap_explicit, pe_explicit = compute_irasa(
            self.sig, self.fs,
            f_range=self.f_range,
            method='welch',
            nperseg=1000
        )
        assert len(freqs_default) == len(freqs_explicit)
        np.testing.assert_array_almost_equal(freqs_default, freqs_explicit)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
