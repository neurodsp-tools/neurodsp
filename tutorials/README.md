# neurodsp Tutorials Index

The neurodsp tutorials are spread out across a series of notebooks, which demonstrate the sequence of analysis you might adopt with using each of the modules. Each tutorial aims to be short and to the point about a particular aspect of idea and module.

- [Filtering](1-Filtering.ipynb)
    - A general tutorial on filtering, filter parameters, and our specific implementation.

- [Instantaneous Amplitude, Frequency, and Phase](2-InstantaneousMeasures.ipynb)
    - Computing instantaneous features, such as amplitude, phase, and frequency, from narrowband-filtered (putative) oscillations.

- [Lagged Coherence](3-LaggedCoherence.ipynb)
    - Our implementation of the lagged coherence algorithm for quantifying the presence of rhythms, see [Fransen et al., 2015, Neuroimage](http://www.sciencedirect.com/science/article/pii/S1053811915004796) for more details.

- [Spectral Analysis](4-SpectralAnalysis.ipynb)
    - Computing and visualizing power spectral density with various methods, Morlet Wavelet Transform, and spectral coefficient of variation (SCV). For parametrizing the resulting spectrum, please see our spectral parametrization toolbox, [FOOOF](https://fooof-tools.github.io/fooof/).

- [Burst Detection](5-BurstDetection.ipynb)
    - Burst detection using the dual threshold algorithm. For a more extensive and time-domain toolbox for detecting contiguous rhythmic cycles, please check out [bycycle](https://voytekresearch.github.io/bycycle/).

- [Sliding Window Matching](6-SlidingWindowMatching.ipynb)
    - Our implementation of the sliding window matching (SWM) algorithm for identifying recurring patterns in a neural signal, like the shape of an oscillatory waveform, see [Gips et al., 2017, J Neuro Methods](http://www.sciencedirect.com/science/article/pii/S0165027016302606)..

- [Simulating Oscillations and Noise](7-SimulatingSignals.ipynb)
    - Simulating 1/f-like signals with various stochastic models, as well as stationary/bursty and non-sinusoidal oscillations.
