"""
sim.py
Simulating oscillators and brown noise
"""

import numpy as np
import pandas as pd
from scipy import signal


def sim_filtered_brown_noise(T, Fs, f_range, N):
    """Simulate a band-pass filtered signal with brown noise
    
    Parameters
    ----------
    T : float
        length of time of simulated oscillation
    Fs : float
        oscillation sampling rate
    f_range : 2-element array (lo,hi)
        frequency range of simulated data
        if None: do not filter
    N : int
        order of filter
        
    Returns
    -------
    brownNf : np.array
        filtered brown noise
    """

    if f_range is None:
        # Do not filter
        # Generate 1/f^2 noise
        brownN = simbrown(int(T*Fs))
        return brownN
    elif f_range[1] is None:
        # Make filter order odd if necessary
        nyq = Fs / 2.
        if N % 2 == 0:
            print('NOTE: Increased high-pass filter order by 1 in order to be odd')
            N += 1
            
        # Generate 1/f^2 noise
        brownN = sim_brown_noise(int(T*Fs+N*2))

        # High pass filter
        taps = signal.firwin(N, f_range[0] / nyq, pass_zero=False)
        brownNf = signal.filtfilt(taps, [1], brownN)
        return brownNf[N:-N]

    else:
        # Bandpass filter
        # Generate 1/f^2 noise
        brownN = simbrown(int(T*Fs+N*2))
        # Filter
        nyq = Fs / 2.
        taps = signal.firwin(N, np.array(f_range) / nyq, pass_zero=False)
        brownNf = signal.filtfilt(taps, [1], brownN)
        return brownNf[N:-N]
    

def sim_brown_noise(N):
    """Simulate a brown noise signal (power law distribution 1/f^2)
    with N samples by cumulative sum of white noise"""
    return np.cumsum(np.random.randn(N))


def sim_oscillator(N_samples_cycle, N_cycles, rdsym=.5):
    """Simulate a band-pass filtered signal with 1/f^2 
    Input suggestions: f_range=(2,None), Fs=1000, N=1001
    
    Parameters
    ----------
    N_samples_cycle : int
        Number of samples in a single cycle
    N_cycles : int
        Number of cycles to simulate
    rdsym : float
        rise-decay symmetry of the oscillator;
        fraction of the period in the rise time;
        =0.5 - symmetric (sine wave)
        <0.5 - shorter rise, longer decay
        >0.5 - longer rise, shorter decay
        
    Returns
    -------
    oscillator : np.array
        oscillating time series
    """
    # Determine number of samples in rise and decay periods
    rise_samples = int(np.round(N_samples_cycle * rdsym))
    decay_samples = N_samples_cycle - rise_samples

    # Make phase array for a single cycle, then repeat it
    pha_one_cycle = np.hstack([np.linspace(0, np.pi, decay_samples+1), np.linspace(-np.pi, 0, rise_samples+1)[1:-1]])
    phase_t = np.tile(pha_one_cycle, N_cycles)
    
    # Transform phase into an oscillator
    oscillator = np.cos(phase_t)
    return oscillator


def sim_noisy_oscillator(freq, T, Fs, rdsym=.5, f_hipass_brown=2, SNR=1):
    """Simulate a band-pass filtered signal with 1/f^2 
    Input suggestions: f_range=(2,None), Fs=1000, N=1001
    
    Parameters
    ----------
    freq : float
        oscillator frequency
    T : float
        signal duration (seconds)
    Fs : float
        signal sampling rate
    f_hipass_brown : float
        frequency (Hz) at which to high-pass-filter
        brown noise
    SNR : float
        ratio of oscillator power to brown noise power
        >1 - oscillator is stronger
        <1 - noise is stronger
        
    Returns
    -------
    signal : np.array
        oscillator with brown noise
    """
    
    # Determine order of highpass filter (3 cycles of f_hipass_brown)
    N = int(3 * Fs / f_hipass_brown)
    if N % 2 == 0:
        N += 1
        
    # Determine length of signal in samples
    N_samples = int(T*Fs)
    
    # Generate filtered brown noise
    brown = sim_filtered_brown_noise(T, Fs, (f_hipass_brown, None), N)
    
    # Generate oscillator
    N_samples_cycle = int(Fs / freq)
    N_cycles = int(np.ceil(N_samples / N_samples_cycle))
    oscillator = sim_oscillator(N_samples_cycle, N_cycles, rdsym=rdsym)
    oscillator = oscillator[:N_samples]
    
    # Normalize brown noise power
    oscillator_power = np.mean(oscillator**2)
    brown_power = np.mean(brown**2)
    brown = np.sqrt(brown**2 * oscillator_power / (brown_power * SNR)) * np.sign(brown)
    # Combine oscillator and noise
    signal = oscillator + brown
    return signal
