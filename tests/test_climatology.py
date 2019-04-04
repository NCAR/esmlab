#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import itertools
import os
import sys

import numpy as np
import pytest
import xarray as xr

from esmlab import compute_ann_mean, compute_mon_anomaly, compute_mon_climatology, compute_mon_mean
from esmlab.datasets import open_dataset

functions = [compute_mon_climatology, compute_mon_anomaly, compute_ann_mean]
decoded = [True, False]
dsets = ['tiny', 'cesm_pop_daily', 'cesm_cice_daily']
params = list(itertools.product(dsets, decoded, functions))


@pytest.mark.skipif(sys.version_info[0] < 3, reason='requires python3')
@pytest.mark.parametrize('ds, decoded, function', params)
def test_compute_climatology(ds, decoded, function):
    dset = open_dataset(ds, decode_times=decoded)
    computed_dset = function(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    for key, value in dset.time.attrs.items():
        assert key in computed_dset.time.attrs
        assert value == computed_dset.time.attrs[key]


@pytest.mark.skipif(sys.version_info[0] < 3, reason='requires python3')
@pytest.mark.parametrize('ds, decoded, function', params)
def test_compute_climatology_drop_time_bound(ds, decoded, function):
    dset = open_dataset(ds, decode_times=decoded)
    dset_time_bound = dset.time.attrs['bounds']
    dset = dset.drop(dset_time_bound)
    del dset.time.attrs['bounds']

    computed_dset = function(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    if function == compute_mon_anomaly:
        assert (dset.time.values == computed_dset.time.values).all()
    for key, value in dset.time.attrs.items():
        assert key in computed_dset.time.attrs
        assert value == computed_dset.time.attrs[key]


@pytest.mark.skipif(sys.version_info[0] < 3, reason='requires python3')
@pytest.mark.parametrize('ds', ['tiny', 'cesm_cice_daily'])
def test_compute_climatology_daisy_chain(ds):
    dset = open_dataset(ds, decode_times=False)

    computed_dset = compute_mon_climatology(dset)
    assert isinstance(computed_dset, xr.Dataset)

    computed_dset2 = compute_mon_anomaly(computed_dset)
    assert isinstance(computed_dset2, xr.Dataset)

    computed_dset3 = compute_ann_mean(computed_dset)
    assert isinstance(computed_dset3, xr.Dataset)

    computed_dset3 = compute_ann_mean(computed_dset2)
    assert isinstance(computed_dset3, xr.Dataset)


def test_mon_climatology_values(dset):
    computed_dset = compute_mon_climatology(dset)
    np.testing.assert_allclose(computed_dset.variable_1.values, 0.5)
    assert computed_dset.time.dtype == dset.time.dtype


def test_mon_anomaly_values(dset):
    computed_dset = compute_mon_anomaly(dset)
    assert isinstance(computed_dset, xr.Dataset)
    a = [-0.5] * 48
    b = [0.5] * 48
    a.extend(b)
    expected = np.array(a)
    np.testing.assert_equal(computed_dset.variable_1.values.ravel(), expected)
    assert computed_dset.time.dtype == dset.time.dtype
    assert (dset.time.values == computed_dset.time.values).all()
    for key, value in dset.time.attrs.items():
        assert key in computed_dset.time.attrs
        assert value == computed_dset.time.attrs[key]


@pytest.mark.skipif(sys.version_info[0] < 3, reason='requires python3')
def test_ann_mean_values(dset):
    weights = np.ones(24)
    computed_dset = compute_ann_mean(dset, weights=weights)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1 / 24, 1 / 24, 1 / 24, 1 / 24], dtype=np.float32)
    np.testing.assert_allclose(
        computed_dset.variable_1.values.ravel().astype(np.float32), expected, rtol=1e-6
    )


@pytest.mark.skipif(sys.version_info[0] < 3, reason='requires python3')
def test_ann_mean_values_missing(dset):
    weights = np.ones(24)
    dset.variable_1.values[0:3, :, :] = np.nan
    computed_dset = compute_ann_mean(dset, weights=weights)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1 / 21, 1 / 21, 1 / 21, 1 / 21], dtype=np.float32)
    np.testing.assert_allclose(
        computed_dset.variable_1.values.ravel().astype(np.float32), expected, rtol=1e-6
    )


@pytest.mark.skipif(sys.version_info[0] < 3, reason='requires python3')
@pytest.mark.parametrize('ds', ['cesm_cam_monthly_full', 'tiny'])
def test_mon_mean(ds):
    dset = open_dataset(ds, decode_times=False, decode_coords=False)
    computed_dset = compute_mon_mean(dset)
    assert isinstance(computed_dset, xr.Dataset)
