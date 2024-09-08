"""Simulation I/O functions that return multiple instances."""

import os
import json
from pathlib import Path

import numpy as np

from neurodsp.sim.signals import Simulations, VariableSimulations, MultiSimulations

###################################################################################################
###################################################################################################

## BASE I/O UTILITIES

def fpath(file_path, file_name):
    """Check and combine a file name and path into a Path object.

    Parameters
    ----------
    file_path : str or None
        Name of the directory.
    file_name : str
        Name of the file.

    Returns
    -------
    Path
        Path object for the full file path.
    """

    return Path(file_path) / file_name if file_path else Path(file_name)


def save_json(file_name, data):
    """Save data to a json file.

    Parameters
    ----------
    file_name : str
        Name of the file to save to.
    data : dict
        Data to save to file.
    """

    with open(file_name, 'w') as json_file:
        json.dump(data, json_file)


def save_jsonlines(file_name, data):
    """Save data to a jsonlines file.

    Parameters
    ----------
    file_name : str
        Name of the file to save to.
    data : list of dict
        Data to save to file.
    """

    with open(file_name, 'w') as jsonlines_file:
        for row in data:
            json.dump(row, jsonlines_file)
            jsonlines_file.write('\n')


def load_json(file_name):
    """Load data from a json file.

    Parameters
    ----------
    file_name : str
        Name of the file to load from.

    Returns
    -------
    data : dict
        Loaded data.
    """

    with open(file_name, 'r') as json_file:
        data = json.load(json_file)

    return data


def load_jsonlines(file_name):
    """Load data from a jsonlines file.

    Parameters
    ----------
    file_name : str
        Name of the file to load from.

    Returns
    -------
    data : list of dict
        Loaded data.
    """

    data = []
    with open(file_name, 'r') as json_file:
        for row in json_file:
            data.append(json.loads(row))

    return data


## SIMULATION OBJECT I/O

def save_sims(sims, label, file_path=None, replace=False):
    """Save simulations.

    Parameters
    ----------
    sims : Simulations or VariableSimulations or MultipleSimulations
        Simulations to save.
    label : str
        Label to attach to the simulation name.
    file_path : str, optional
        Directory to save to.
    replace : bool, optional, default: False
        Whether to replace any existing saved files with the same name.
    """

    assert '_' not in label, 'Cannot have underscores in simulation label.'

    save_path_items = ['sim_unknown' if not sims.sim_func else sims.sim_func]
    if isinstance(sims, (VariableSimulations, MultiSimulations)):
        if sims.component:
            save_path_items.append(sims.component)
        if sims.update:
            save_path_items.append(sims.update)
    save_path_items.append(label)
    save_path = '_'.join(save_path_items)

    save_folder = fpath(file_path, save_path)

    if os.path.exists(save_folder):
        if not replace:
            raise ValueError('Simulation files already exist.')
    else:
        os.mkdir(save_folder)

    if isinstance(sims, MultiSimulations):
        for ind, csims in enumerate(sims.signals):
            save_sims(csims, 'set' + str(ind), file_path=save_folder, replace=True)

    else:
        np.save(save_folder / 'signals', sims.signals)
        if isinstance(sims, VariableSimulations):
            save_jsonlines(save_folder / 'params.jsonlines', sims.params)
        elif isinstance(sims, Simulations):
            save_json(save_folder / 'params.json', sims.params)


def load_sims(load_name, file_path=None):
    """Load simulations.

    Parameters
    ----------
    load_name : str
        The name or label of the simulations to load.
        If not the full file name, this string can be the label the simulations were saved with.
    file_path : str, optional
        Directory to load from.

    Returns
    -------
    sims : Simulations or VariableSimulations or MultipleSimulations
        Loaded simulations.
    """

    if '_' not in load_name:
        matches = [el for el in os.listdir(file_path) if load_name in el]
        assert len(matches) > 0, 'No matches found for requested simulation label.'
        assert len(matches) == 1, 'Multiple matches found for requested simulation label.'
        load_name = matches[0]

    splits = load_name.split('_')
    sim_func = '_'.join(splits[0:2]) if splits[1] != 'unknown' else None

    update, component = None, None
    if len(splits) > 3:
        splits = splits[2:-1]
        update = splits.pop()
        component = '_'.join(splits) if splits else None

    load_folder = fpath(file_path, load_name)
    load_files = [file for file in os.listdir(load_folder) if file[0] != '.']

    if 'signals.npy' not in load_files:

        msims = [load_sims(load_file, load_folder) for load_file in load_files]
        sims = MultiSimulations(msims, None, sim_func, update, component)

    else:

        sigs = np.load(load_folder / 'signals.npy')

        if 'params.json' in load_files:
            params = load_json(load_folder / 'params.json')
            sims = Simulations(sigs, params, sim_func)

        elif 'params.jsonlines' in load_files:
            params = load_jsonlines(load_folder / 'params.jsonlines')
            sims = VariableSimulations(sigs, params, sim_func, update, component)

    return sims
