"""Test simulation functions.

Code to make true data files:

import numpy as np
import os
import neurodsp
from neurodsp.sim import *

n_seconds = 10
fs = 1000
freq = 6
rdsym = .5
f_range_filter = (2, None)
filter_order = 1501
exponent = 2
ratio_osc_var = 1

np.random.seed(0)
noise = sim_filtered_noise(n_seconds, fs, f_range_filter, filter_order, exponent=exponent)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/noise_filt.npy', noise)

np.random.seed(0)
osc = sim_oscillation(n_seconds, fs, freq, rdsym=rdsym)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_osc.npy', osc)

np.random.seed(0)
osc = sim_noisy_oscillation(n_seconds, fs, freq, exponent=exponent,
                           rdsym=rdsym, f_range_filter=f_range_filter,
                           ratio_osc_var=ratio_osc_var, filter_order=filter_order)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_noisy_osc.npy', osc)

np.random.seed(0)
osc = sim_bursty_oscillation(n_seconds, fs, freq, rdsym=rdsym)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_bursty_osc.npy', osc)

np.random.seed(0)
osc = sim_noisy_bursty_oscillation(n_seconds, fs, freq, rdsym=rdsym, exponent=exponent,
                                  f_range_filter=f_range_filter, ratio_osc_var=ratio_osc_var,
                                  prob_enter_burst=.2, prob_leave_burst=.2,
                                  cycle_features=None, return_components=False,
                                  return_cycle_df=False)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_noisy_bursty_osc.npy', osc)

np.random.seed(0)
syn_noise = sim_synaptic_noise(2, 1000, 1000, 2, 1., 0.002, 2)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_synaptic_noise.npy', syn_noise)

np.random.seed(0)
ou_noise = sim_ou_process(2, 1000, 1., 0., 5.)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_ou_process.npy', ou_noise)

np.random.seed(0)
jittered_osc = sim_jittered_oscillation(2, 1000, 20, 0.00, ('gaussian',0.01))
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_jittered_oscillation.npy', jittered_osc)

np.random.seed(0)
powerlaw = sim_variable_powerlaw(60, 1000, -2.25)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_variable_powerlaw.npy', powerlaw)

"""