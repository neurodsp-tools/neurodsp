"""Test plot utilities."""

import os
import tempfile

from neurodsp.plts.utils import *

###################################################################################################
###################################################################################################

def test_check_ax():

    # Check running will None Input
    ax = check_ax(None)

    # Check running with pre-created axis
    _, ax = plt.subplots()
    nax = check_ax(ax)
    assert nax == ax

    # Check creating figure of a particular size
    figsize = [5, 5]
    ax = check_ax(None, figsize=figsize)
    fig = plt.gcf()
    assert list(fig.get_size_inches()) == figsize

def test_savefig():

    @savefig
    def example_plot():
        plt.plot([1, 2], [3, 4])

    with tempfile.NamedTemporaryFile(mode='w+') as file:
        example_plot(save_fig=True, file_name=file.name)
        assert os.path.exists(file.name)
