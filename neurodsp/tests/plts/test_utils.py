"""Tests for neurodsp.plts.utils."""

import os

import matplotlib as mpl

from neurodsp.tests.settings import TEST_PLOTS_PATH

from neurodsp.plts.utils import *

###################################################################################################
###################################################################################################

def test_subset_kwargs():

    kwargs = {'xlim' : [0, 10], 'ylim' : [2, 5],
              'title_fontsize' : 24, 'title_fontweight': 'bold'}

    kwargs1, subset1 = subset_kwargs(kwargs, 'lim')
    assert list(kwargs1.keys()) == ['title_fontsize', 'title_fontweight']
    assert list(subset1.keys()) == ['xlim', 'ylim']

    kwargs2, subset2 = subset_kwargs(kwargs, 'title')
    assert list(kwargs2.keys()) == ['xlim', 'ylim']
    assert list(subset2.keys()) == ['title_fontsize', 'title_fontweight']

def test_check_ax():

    # Check running with None Input
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

    # Test defaults to saving given file path & name
    example_plot(file_path=TEST_PLOTS_PATH, file_name='test_savefig1.pdf')
    assert os.path.exists(os.path.join(TEST_PLOTS_PATH, 'test_savefig1.pdf'))

    # Test works the same when explicitly given `save_fig`
    example_plot(save_fig=True, file_path=TEST_PLOTS_PATH, file_name='test_savefig2.pdf')
    assert os.path.exists(os.path.join(TEST_PLOTS_PATH, 'test_savefig2.pdf'))

    # Test giving additional save kwargs
    example_plot(file_path=TEST_PLOTS_PATH, file_name='test_savefig3.pdf',
                 save_kwargs={'facecolor' : 'red'})
    assert os.path.exists(os.path.join(TEST_PLOTS_PATH, 'test_savefig3.pdf'))

    # Test does not save when `save_fig` set to False
    example_plot(save_fig=False, file_path=TEST_PLOTS_PATH, file_name='test_savefig_nope.pdf')
    assert not os.path.exists(os.path.join(TEST_PLOTS_PATH, 'test_savefig_nope.pdf'))

def test_save_figure():

    plt.plot([1, 2], [3, 4])
    save_figure(file_name='test_save_figure.pdf', file_path=TEST_PLOTS_PATH)
    assert os.path.exists(os.path.join(TEST_PLOTS_PATH, 'test_save_figure.pdf'))

def test_make_axes():

    axes = make_axes(2, 2)
    assert axes.shape == (2, 2)
    assert isinstance(axes[0, 0], mpl.axes._axes.Axes)
