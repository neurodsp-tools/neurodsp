"""
test_sim.py
Test simulation of noisy oscillators
"""

import numpy as np
import os
import neurodsp


def test_sim_consistent():
    """
    Confirm consistency in simulation

    Previous code run:
    import numpy as np
    import neurodsp
    import os
    np.random.seed(0)

    # Simulate brown noise
    brown = neurodsp.sim_brown_noise(1000)
    np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_brown.npy', brown)

    # Simulate highpass-filtered brown noise
    brown_filt = neurodsp.sim_filtered_brown_noise(10, 1000, (2, None), 1501)
    np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_brown_filt.npy', brown_filt)

    # Simulate oscillator
    osc = neurodsp.sim_oscillator(100, 100, rdsym=.3)
    np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_osc.npy', osc)

    # Simulate noisy oscillator
    signal = neurodsp.sim_noisy_oscillator(6, 2, 1000, rdsym=.5, f_hipass_brown=2, SNR=2)
    np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_noisy_osc.npy', signal)

    # Simulate bursty oscillator
    bursty_osc = neurodsp.sim_bursty_oscillator(6, 10, 1000)
    np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_bursty_osc.npy', bursty_osc)

    # Simulate noisy bursty oscillator
    bursty_signal = neurodsp.sim_noisy_bursty_oscillator(6, 2, 1000, rdsym=.5, f_hipass_brown=2, SNR=2)
    np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_noisy_bursty_osc.npy', bursty_signal)
    """

    # Simulate noise and oscillation
    np.random.seed(0)
    brown = neurodsp.sim_brown_noise(1000)
    brown_filt = neurodsp.sim_filtered_brown_noise(10, 1000, (2, None), 1501)
    osc = neurodsp.sim_oscillator(100, 100, rdsym=.3)
    noisy_osc = neurodsp.sim_noisy_oscillator(
        6, 2, 1000, rdsym=.5, f_hipass_brown=2, SNR=2)
    bursty_osc = neurodsp.sim_bursty_oscillator(6, 10, 1000)
    bursty_noisy_osc = neurodsp.sim_noisy_bursty_oscillator(
        6, 2, 1000, rdsym=.5, f_hipass_brown=2, SNR=2)

    # Load noise and oscillation
    brown_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_brown.npy')
    brown_filt_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_brown_filt.npy')
    osc_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_osc.npy')
    noisy_osc_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_noisy_osc.npy')
    bursty_osc_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_bursty_osc.npy')
    bursty_noisy_osc_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_noisy_bursty_osc.npy')

    # Test consistency between all signals
    assert np.allclose(np.sum(np.abs(brown - brown_true)), 0, atol=10 ** -5)
    assert np.allclose(
        np.sum(np.abs(brown_filt - brown_filt_true)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(osc - osc_true)), 0, atol=10 ** -5)
    assert np.allclose(
        np.sum(np.abs(noisy_osc - noisy_osc_true)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(bursty_osc - bursty_osc_true)), 0, atol=10 ** -5)
    assert np.allclose(
        np.sum(np.abs(bursty_noisy_osc - bursty_noisy_osc_true)), 0, atol=10 ** -5)
