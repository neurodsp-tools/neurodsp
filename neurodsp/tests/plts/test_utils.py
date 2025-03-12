"""Tests for neurodsp.plts.utils."""

from pytest import raises

import os
import itertools

import numpy as np
import matplotlib as mpl

from neurodsp.tests.tsettings import TEST_PLOTS_PATH

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

def test_check_ax_3d():

    # Check running with None Input
    ax = check_ax(None)

    # Check error if given a non 3D axis
    with raises(ValueError):
        _, ax = plt.subplots()
        nax = check_ax_3d(ax)

    # Check running with pre-created axis
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    nax = check_ax(ax)
    assert nax == ax

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

def test_prepare_multi_plot():

    xs1 = np.array([1, 2, 3])
    ys1 = np.array([1, 2, 3])
    labels1 = None
    colors1 = None

    # 1 input
    xs1o, ys1o, labels1o, colors1o = prepare_multi_plot(xs1, ys1, labels1, colors1)
    assert isinstance(xs1o, itertools.repeat)
    assert isinstance(ys1o, list)
    assert isinstance(labels1o, itertools.repeat)
    assert isinstance(colors1o, itertools.repeat)

    # multiple inputs
    xs2 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    ys2 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    labels2 = ['A', 'B']
    colors2 = ['blue', 'red']
    xs2o, ys2o, labels2o, colors2o = prepare_multi_plot(xs2, ys2, labels2, colors2)
    assert isinstance(xs2o, list)
    assert isinstance(ys2o, list)
    assert isinstance(labels2o, list)
    assert isinstance(colors2o, itertools.cycle)
