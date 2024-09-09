"""
Managing Simulation Parameters
==============================

Manage, update, and iterate across simulation parameters.
"""

from neurodsp.sim.update import create_updater, create_sampler
from neurodsp.sim.params import SimParams, SimIters, SimSamplers

###################################################################################################
# Managing Simulations Parameters
# -------------------------------
#
# The :class:`~.SimParams` object can be used to manage a set of simulation parameters.
#

###################################################################################################

# Initialize object, with base parameters
sim_params = SimParams(n_seconds=5, fs=250)

# Check the base parameters in the SimParams object
sim_params.base

###################################################################################################
#
# A defined SimParams object with base parameters can be used to create a full set of simulation
# parameters by specifying additional parameters to add to the base parameters.
#

###################################################################################################

# Create a set of simulation parameters
sim_params.make_params({'exponent' : -1})

###################################################################################################
#
# The object can also be used to 'register' (:func:`~.SimParams.register`) a set of
# simulation parameters, meaning they can be defined and stored in the object,
# with an associated label to access them.
#

###################################################################################################

# Register a set of simulation parameters
sim_params.register('ap', {'exponent' : -1})

# Check the registered simulation definition
sim_params['ap']

###################################################################################################
#
# The SimParams object can also be updated, for example, clearing previous simulation parameters,
# updating base parameters, and/or updating previously registered simulation definitions.
#

###################################################################################################

# Clear the current set of parameter definitions
sim_params.clear()

# Update the base definition
sim_params.update_base(n_seconds=10)

# Check the updated base parameters
sim_params.base

###################################################################################################
#
# The SimParams object can also be used to manage multiple different simulation parameter
# definitions, for example for different functions, which share the same base parameters.
#
# To manage multiple parameters, they can all be registered to the object.
# For convenience, multiple definitions can be registered together with the
# (:func:`~.SimParams.register_group`) method.
#

###################################################################################################

# Register a group of parameter definitions
sim_params.register_group(
    {'ap' : {'exponent' : -1},
     'osc' : {'freq' : 10}})

###################################################################################################

# Check the set of labels and parameters defined on the object
print(sim_params.labels)
print(sim_params.params)

###################################################################################################

# Check the simulation parameters for the different labels
print(sim_params['ap'])
print(sim_params['osc'])

###################################################################################################
# Iterating Across Simulations Parameters
# ---------------------------------------
#
# One application of interest for managing simulation parameters may be to iterate
# across parameter values.
#
# To do so, the :class:`~.SimIters` class can be used.
#

###################################################################################################

# Initialize base set of simulation parameters
sim_iters = SimIters(n_seconds=5, fs=250)

# Check the base parameters of the SimIters object
sim_iters.base

###################################################################################################

# Re-initialize a SimIters object, exporting from existing SimParams object
sim_iters = sim_params.to_iters()

###################################################################################################
#
# Similar to the SimParams object, the SimIter object can be used to make simulation iterators.
#

###################################################################################################

# Make a parameter iterator from the SimIter object
exp_iter = sim_iters.make_iter('ap', 'exponent', [-2, -1, 0])

# Use the iterator to step across parameters
for params in exp_iter:
    print(params)

###################################################################################################
#
# Just as before, we can 'register' an iterator definition on the SimIter object.
#

###################################################################################################

# Register an iterator on the SimIter object
sim_iters.register_iter('exp_iter', 'ap', 'exponent', [-2, -1, 0])

# Use the iterator from the SimIter object to step across parameters
for params in sim_iters['exp_iter']:
    print(params)

###################################################################################################
#
# Just like the SimParams object, the SimIter object can be cleared, updated, etc.
#
# It can also be used to register a group of iterators, which will share the same base parameters.
#

###################################################################################################

# Clear the current object
sim_iters.clear()

# Register a group of iterators
sim_iters.register_group_iters([
    ['exp_iter', 'ap', 'exponent', [-2, -1, 0]],
    ['osc_iter', 'osc', 'freq', [10, 20, 30]]])

###################################################################################################

# Check the labels for the defined iterators, and the iterators
print(sim_iters.labels)
print(sim_iters.iters)

# Check a set of iterated parameters from the SimIter object
for params in sim_iters['osc_iter']:
    print(params)

###################################################################################################
# Defining Parameters Updates
# ---------------------------
#
# For the next application, we will explore defining sets of parameters to sample from.
#
# To do so, we first need to explore some functionality for defining which parameters to
# update, and how to sample parameter values from a specified set of objects.
#
# To start with, we can use the :func:`~.create_updater` function to create a helper
# function to update parameters.
#

###################################################################################################

# Define a set of parameters
params1 = {'n_seconds' : 5, 'fs' : 250, 'exponent' : None}

# Create an update object for the exponent parameter
exp_updater = create_updater('exponent')

# Use the exponent updater
exp_updater(params1, -1)

###################################################################################################
#
# An updater can also be used to update parameters defined within specified components.
#

###################################################################################################

# Define another set of parameters, with multiple components
params2 = {'n_seconds' : 5, 'fs' : 250,
           'components' : {'sim_powerlaw' : {'exponent' : None},
                           'sim_oscillation' : {'freq' : 10}}}

# Create an updater for the exponent, within the components
exp_comp_updater = create_updater('exponent', 'sim_powerlaw')

# Use the exponent updater
exp_comp_updater(params2, -1)

###################################################################################################
#
# Next, we can define a way to sample parameter values.
#
# To do so, we can use the :func:`~.create_sampler` function.
#

###################################################################################################

# Create a sampler for a set of exponent values
exp_sampler = create_sampler([-2, -1, 0])

# Sample some values from the exponent sampler
for ind in range(3):
    print(next(exp_sampler))

###################################################################################################
#
# From the above, we can combine updaters and samplers to create definitions of how
# to sample full parameter definitions.
#

###################################################################################################

# Define a combined updater and sampler for exponent values
exp_upd_sampler = {create_updater('exponent') : create_sampler([-2, -1, 0])}

###################################################################################################
# Sampling Simulations Parameters
# -------------------------------
#
# To manage sampling simulation parameters, we can use the :class:`~.SimSamplers` class.
#

###################################################################################################

# Initialize simulation samplers, from pre-initialized SimParams object
sim_samplers = sim_params.to_samplers(n_samples=3)

###################################################################################################
#
# Just as before, the SimSamplers object can be used to make samplers.
#

###################################################################################################

# Make a parameter sampler from the SimSamplers object
exp_sampler = sim_samplers.make_sampler('ap', exp_upd_sampler)

# Use the exponent sampler to check
for samp_params in exp_sampler:
    print(samp_params)

###################################################################################################
#
# As before, we can also register a sampler on the object.
#

###################################################################################################

# Register a sampler definition on the SimSamplers object
sim_samplers.register_sampler('exp_sampler', 'ap', exp_upd_sampler)

# Check some example sampled parameter values
for samp_params in sim_samplers['exp_sampler']:
    print(samp_params)

###################################################################################################
#
# The object can also be cleared, updated, etc, just as the previous objects.
#

###################################################################################################

# Clear the previously defined simulation samplers
sim_samplers.clear()

# Define a new definition to sample parameter values
osc_upd_sampler = {create_updater('freq') : create_sampler([10, 20, 30])}

# Register a group of samplers to the object
sim_samplers.register_group_samplers([
    ['exp_sampler', 'ap', exp_upd_sampler],
    ['osc_sampler', 'osc', osc_upd_sampler],
])

###################################################################################################

# Check the labels and defined samplers on the object
print(sim_samplers.labels)
print(sim_samplers.samplers)

# Check example sampled parameter values
for samp_params in sim_samplers['osc_sampler']:
    print(samp_params)
