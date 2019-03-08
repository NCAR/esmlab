#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pytest
import xarray as xr

from esmlab.climatology import compute_ann_mean, compute_mon_anomaly, compute_mon_climatology
from esmlab.datasets import open_dataset


@pytest.mark.parametrize('ds', ['tiny', 'cesm_cice_daily'])
def test_compute_climatology_multi(ds):
    dset = open_dataset(ds, decode_times=False)
    print(dset)
    computed_dset = compute_mon_climatology(dset)
    print(computed_dset)
    print()
    assert isinstance(computed_dset, xr.Dataset)
    computed_dset = compute_ann_mean(dset)
    print(computed_dset)
    print()
    assert isinstance(computed_dset, xr.Dataset)
    computed_dset = compute_mon_anomaly(dset)
    print(computed_dset)
    print()
    assert isinstance(computed_dset, xr.Dataset)


def test_compute_mon_climatology(dset):
    computed_dset = compute_mon_climatology(dset)
    np.testing.assert_equal(computed_dset.variable_1.values, 0.5)


def test_compute_mon_climatology_times_decoded(dset):
    computed_dset = compute_mon_climatology(dset)
    np.testing.assert_equal(computed_dset.variable_1.values, 0.5)


def test_compute_mon_anomaly(dset):
    computed_dset = compute_mon_anomaly(dset)
    assert isinstance(computed_dset, xr.Dataset)
    a = [-0.5] * 48
    b = [0.5] * 48
    a.extend(b)
    expected = np.array(a)
    np.testing.assert_equal(computed_dset.variable_1.values.ravel(), expected)


def test_compute_ann_mean(dset):
    computed_dset = compute_ann_mean(dset)
    assert isinstance(computed_dset, xr.Dataset)
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_equal(computed_dset.variable_1.values.ravel(), expected)
