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

    computed_dset = compute_mon_climatology(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    for key, value in dset.time.attrs.items():
        assert key in computed_dset.time.attrs
        assert value == computed_dset.time.attrs[key]

    computed_dset = compute_ann_mean(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    for key, value in dset.time.attrs.items():
        assert key in computed_dset.time.attrs
        assert value == computed_dset.time.attrs[key]

    computed_dset = compute_mon_anomaly(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    assert (dset.time.values == computed_dset.time.values).all()
    for key, value in dset.time.attrs.items():
        assert key in computed_dset.time.attrs
        assert value == computed_dset.time.attrs[key]


@pytest.mark.parametrize('ds', ['tiny', 'cesm_cice_daily'])
def test_compute_climatology_multi_decoded(ds):
    dset = open_dataset(ds, decode_times=True)

    computed_dset = compute_mon_climatology(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    for key, value in dset.time.attrs.items():
        assert key in computed_dset.time.attrs
        assert value == computed_dset.time.attrs[key]

    computed_dset = compute_ann_mean(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    for key, value in dset.time.attrs.items():
        assert key in computed_dset.time.attrs
        assert value == computed_dset.time.attrs[key]

    computed_dset = compute_mon_anomaly(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    assert (dset.time.values == computed_dset.time.values).all()
    for key, value in dset.time.attrs.items():
        assert key in computed_dset.time.attrs
        assert value == computed_dset.time.attrs[key]


@pytest.mark.parametrize('ds', ['tiny', 'cesm_cice_daily'])
def test_compute_climatology_multi_drop_time_bound_decoded(ds):
    dset = open_dataset(ds, decode_times=True)
    dset_time_bound = dset.time.attrs['bounds']
    dset = dset.drop(dset_time_bound)
    del dset.time.attrs['bounds']

    computed_dset = compute_mon_climatology(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    for key, value in dset.time.attrs.items():
        assert key in computed_dset.time.attrs
        assert value == computed_dset.time.attrs[key]

    computed_dset = compute_ann_mean(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    for key, value in dset.time.attrs.items():
        assert key in computed_dset.time.attrs
        assert value == computed_dset.time.attrs[key]

    computed_dset = compute_mon_anomaly(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    assert (dset.time.values == computed_dset.time.values).all()
    for key, value in dset.time.attrs.items():
        assert key in computed_dset.time.attrs
        assert value == computed_dset.time.attrs[key]


@pytest.mark.parametrize('ds', ['tiny', 'cesm_cice_daily'])
def test_compute_climatology_multi_drop_time_bound(ds):
    dset = open_dataset(ds, decode_times=False)
    dset_time_bound = dset.time.attrs['bounds']
    dset = dset.drop(dset_time_bound)
    del dset.time.attrs['bounds']

    computed_dset = compute_mon_climatology(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    for key, value in dset.time.attrs.items():
        assert key in computed_dset.time.attrs
        assert value == computed_dset.time.attrs[key]

    computed_dset = compute_ann_mean(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    for key, value in dset.time.attrs.items():
        assert key in computed_dset.time.attrs
        assert value == computed_dset.time.attrs[key]

    computed_dset = compute_mon_anomaly(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    assert (dset.time.values == computed_dset.time.values).all()
    for key, value in dset.time.attrs.items():
        assert key in computed_dset.time.attrs
        assert value == computed_dset.time.attrs[key]


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


def test_compute_mon_climatology_values(dset):
    computed_dset = compute_mon_climatology(dset)
    np.testing.assert_equal(computed_dset.variable_1.values, 0.5)
    assert computed_dset.time.dtype == dset.time.dtype


def test_compute_mon_anomaly_values(dset):
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


def test_compute_ann_mean_values(dset):
    computed_dset = compute_ann_mean(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_equal(computed_dset.variable_1.values.ravel(), expected)


def test_compute_ann_mean_values_missing(dset):
    dset.variable_1.values[0:3, :, :] = np.nan
    computed_dset = compute_ann_mean(dset)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_equal(computed_dset.variable_1.values.ravel(), expected)
