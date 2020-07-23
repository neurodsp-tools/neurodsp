"""Utilities for accessing the simulation functions and retrieving information about them."""

from inspect import getmembers, isfunction

###################################################################################################
###################################################################################################

SIM_MODULES = ['periodic', 'aperiodic', 'cycles', 'transients', 'combined']

def get_sim_funcs(module_name):
    """Get the available sim functions from a specified sub-module.

    Parameters
    ----------
    module_name : {'periodic', 'aperiodic', 'transients', 'combined'}
        Simulation sub-module to get sim functions from.

    Returns
    -------
    funcs : dictionary
        A dictionary containing the available sim functions from the requested sub-module.
    """

    # Note: imports done within function to avoid circular import
    from neurodsp.sim import periodic, aperiodic, transients, combined, cycles

    if module_name in SIM_MODULES:
        module = eval(module_name)
    else:
        raise ValueError('Requested sim module not understood.')

    funcs = {name : func for name, func in getmembers(module, isfunction) \
        if name[0:4] == 'sim_' and func.__module__.split('.')[-1] == module.__name__.split('.')[-1]}

    return funcs


def get_sim_names(module_name):
    """Get the names of the available sim functions from a specified sub-module.

    Parameters
    ----------
    module_name : {'periodic', 'aperiodic', 'transients', 'combined'}
        Simulation sub-module to get sim functions from.

    Returns
    -------
    list of str
        The names of the available functions in the requested sub-module.
    """

    return list(get_sim_funcs(module_name).keys())


def get_sim_func(function_name, modules=SIM_MODULES):
    """Get a specified sim function.

    Parameters
    ----------
    function_name : str
        Name of the sim function to retrieve.
    modules : list of str, optional
        Which sim modules to look for the function in.

    Returns
    -------
    func : callable
        Requested sim function.
    """

    for module in modules:
        try:
            func = get_sim_funcs(module)[function_name]
            break
        except KeyError:
            continue

    else:
        raise ValueError('Requested simulation function not found.') from None

    return func
