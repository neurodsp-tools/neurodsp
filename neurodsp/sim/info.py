"""Utilities for accessing the simulation functions and retrieving information about them."""

from inspect import getmembers, isfunction

from neurodsp.utils.checks import check_param_options

###################################################################################################
###################################################################################################

SIM_MODULES = ['periodic', 'aperiodic', 'cycles', 'transients', 'combined']

def get_sim_funcs(module):
    """Get the available sim functions from a specified sub-module.

    Parameters
    ----------
    module : {'periodic', 'aperiodic', 'cycles', 'transients', 'combined'}
        Simulation sub-module to get sim functions from.

    Returns
    -------
    functions : dictionary
        A dictionary containing the available sim functions from the requested sub-module.
    """

    check_param_options(module, 'module', SIM_MODULES)

    # Note: imports done within function to avoid circular import
    from neurodsp.sim import periodic, aperiodic, transients, combined, cycles

    module = eval(module)

    module_name = module.__name__.split('.')[-1]
    functions = {name : function for name, function in getmembers(module, isfunction) \
        if name[0:4] == 'sim_' and function.__module__.split('.')[-1] == module_name}

    return functions


def get_sim_names(module):
    """Get the names of the available sim functions from a specified sub-module.

    Parameters
    ----------
    module : {'periodic', 'aperiodic', 'transients', 'combined'}
        Simulation sub-module to get sim functions from.

    Returns
    -------
    list of str
        The names of the available functions in the requested sub-module.
    """

    return list(get_sim_funcs(module).keys())


def get_sim_func(function, modules=SIM_MODULES):
    """Get a specified sim function.

    Parameters
    ----------
    function : str or callabe
        Name of the sim function to retrieve.
        If callable, returns input.
        If string searches for corresponding callable sim function.
    modules : list of str, optional
        Which sim modules to look for the function in.

    Returns
    -------
    function : callable
        Requested sim function.
    """

    if callable(function):
        return function

    for module in modules:
        try:
            function = get_sim_funcs(module)[function]
            break
        except KeyError:
            continue

    else:
        raise ValueError('Requested simulation function not found.') from None

    return function


def get_sim_func_name(function):
    """Get the name of a simulation function.

    Parameters
    ----------
    function : str or callabe
        Function to get name for.

    Returns
    -------
    name : str
        Name of the function.
    """

    name = function.__name__ if callable(function) else function

    return name
