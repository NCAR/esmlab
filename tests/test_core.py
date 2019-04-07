#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import itertools
import os
from collections import OrderedDict

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas.testing import assert_frame_equal

import esmlab
from esmlab.datasets import open_dataset


def assert_both_frames_equal(x, y):
    np.testing.assert_allclose(x, y, rtol=1e-07, atol=1e-1)


def test_esmlab_accessor():
    ds = xr.Dataset(
        {
            'temp': xr.DataArray(
                [1, 2],
                dims=['time'],
                coords={'time': pd.date_range(start='2000', periods=2, freq='1D')},
            )
        }
    )
    attrs = {'calendar': 'noleap', 'units': 'days since 2000-01-01 00:00:00'}
    ds.time.attrs = attrs
    esm = ds.esmlab.set_time(time_coord_name='time')
    # Time and Time bound Attributes
    expected = dict(esm.time_attrs)
    attrs['bounds'] = None
    assert expected == attrs
    assert esm.time_bound_attrs == {}

    assert esm.variables == ['temp']
    assert esm.static_variables == []

    # Time bound diff
    expected = xr.ones_like(ds.time, dtype='float64')
    xr.testing.assert_equal(expected, esm.time_bound_diff)

    # Compute time var
    with pytest.raises(ValueError):
        esm.compute_time_var(midpoint=True, year_offset=2100)

    # Decode arbitrary time value
    with pytest.raises(ValueError):
        esm.decode_arbitrary_time(ds.time.data[0], units=attrs['units'], calendar=attrs['calendar'])

    res = esm.decode_arbitrary_time(
        np.array([30]), units=attrs['units'], calendar=attrs['calendar']
    )
    assert res[0] == cftime.DatetimeNoLeap(2000, 1, 31, 0, 0, 0, 0, 0, 31)

    data = xr.DataArray(
        [1, 2],
        dims=['time'],
        coords={'time': pd.date_range(start='2000', freq='1D', periods=2)},
        attrs={'calendar': 'standard', 'units': 'days since 2001-01-01 00:00:00'},
        name='rand',
    ).to_dataset()

    data['time'] = xr.cftime_range(start='2000', freq='1D', periods=2)

    with pytest.raises(ValueError):
        data.esmlab.set_time().get_time_decoded()

    with pytest.raises(ValueError):
        data.esmlab.set_time().get_time_undecoded()

    data = xr.DataArray(
        [[1, 2], [7, 8]], dims=['x', 'y'], coords={'x': [1, 2], 'y': [2, 3]}, name='rand'
    ).to_dataset()
    with pytest.raises(ValueError):
        data.esmlab.set_time('time-bound-coord')


def test_time_bound_var(dset, time_coord_name='time'):
    esm = dset.esmlab.set_time(time_coord_name=time_coord_name)
    results = esm.tb_name, esm.tb_dim
    expected = ('time_bound', 'd2')
    assert results == expected


def test_time_year_to_midyeardate(dset, time_coord_name='time'):
    assert isinstance(dset[time_coord_name].values[0], np.number)
    dset = dset.esmlab.set_time(time_coord_name=time_coord_name).time_year_to_midyeardate()
    assert isinstance(dset[time_coord_name].values[0], cftime.datetime)


def test_get_time_attrs(dset, time_coord_name='time'):
    expected = {
        'units': 'days since 0001-01-01 00:00:00',
        'calendar': 'noleap',
        'bounds': 'time_bound',
    }
    esm = dset.esmlab.set_time(time_coord_name=time_coord_name)
    results = esm.time_attrs
    assert results == expected


def test_compute_time_var(dset, time_coord_name='time'):
    idx = dset.indexes[time_coord_name]
    assert isinstance(idx, pd.core.indexes.numeric.Index)
    esm = dset.esmlab.set_time(time_coord_name=time_coord_name)
    results = esm.get_time_decoded()
    assert isinstance(results, xr.DataArray)


def test_uncompute_time_var(dset, time_coord_name='time'):
    esm = dset.esmlab.set_time(time_coord_name=time_coord_name)
    ds = esm.compute_time_var()
    assert ds[time_coord_name].dtype == np.dtype('O')
    dset_with_uncomputed_time = esm.uncompute_time_var()
    assert np.issubdtype(dset_with_uncomputed_time[time_coord_name].dtype, np.number)


# For some strange reason, this case fails when using pytest parametrization
def test_sel_time_(dset):
    esm = dset.esmlab.set_time()
    dset = esm.sel_time(indexer_val=slice('1850-01-01', '1850-12-31'), year_offset=1850)
    assert len(dset.time) == 12


@pytest.mark.parametrize(
    'ds_name, decoded, variables, time_coord_name',
    [
        ('tiny', True, ['variable_1', 'variable_2'], 'time'),
        ('tiny', False, ['variable_1', 'variable_2'], 'time'),
        ('cmip5_pr_amon_csiro', False, ['pr'], 'time'),
        ('cmip5_pr_amon_csiro', True, ['pr'], 'time'),
        ('cesm_cice_daily', False, ['aicen_d'], 'time'),
        ('cesm_pop_daily', True, ['FG_CO2_2'], 'time'),
    ],
)
def test_mon_climatology(ds_name, decoded, variables, time_coord_name):
    ds = esmlab.datasets.open_dataset(ds_name, decode_times=decoded)
    esm = ds.esmlab.set_time(time_coord_name=time_coord_name)
    computed_dset = esmlab.climatology(ds, freq='mon')
    esmlab_res = computed_dset.drop(esm.static_variables).to_dataframe()
    esmlab_res = esmlab_res.groupby('month').mean()[variables]

    df = (
        esm._ds_time_computed.drop(esm.static_variables)
        .to_dataframe()
        .reset_index(level=[esm.time_coord_name])
    )
    df[esm.time_coord_name] = df[esm.time_coord_name].values.astype('datetime64[ns]')
    pd_res = df.groupby(df[esm.time_coord_name].dt.month).mean()
    pd_res.index.name = 'month'
    pd_res = pd_res[variables]

    assert_both_frames_equal(esmlab_res, pd_res)

    assert computed_dset[esm.time_coord_name].dtype == ds[esm.time_coord_name].dtype
    for key, value in ds[esm.time_coord_name].attrs.items():
        assert key in computed_dset[esm.time_coord_name].attrs
        assert value == computed_dset[esm.time_coord_name].attrs[key]


@pytest.mark.parametrize(
    'ds_name, decoded, variables, time_coord_name, time_bound_name',
    [
        ('tiny', True, ['variable_1', 'variable_2'], 'time', 'bounds'),
        ('tiny', False, ['variable_1', 'variable_2'], 'time', 'bounds'),
        ('cmip5_pr_amon_csiro', False, ['pr'], 'time', 'bounds'),
        ('cmip5_pr_amon_csiro', True, ['pr'], 'time', 'bounds'),
        ('cesm_cice_daily', False, ['aicen_d'], 'time', 'bounds'),
        ('cesm_pop_daily', True, ['FG_CO2_2'], 'time', 'bounds'),
    ],
)
def test_mon_climatology_drop_time_bounds(
    ds_name, decoded, variables, time_coord_name, time_bound_name
):
    ds = esmlab.datasets.open_dataset(ds_name, decode_times=decoded)
    ds_time_bound = ds[time_coord_name].attrs[time_bound_name]
    ds = ds.drop(ds_time_bound)
    del ds[time_coord_name].attrs[time_bound_name]
    esm = ds.esmlab.set_time(time_coord_name=time_coord_name)
    computed_dset = esmlab.climatology(ds, freq='mon')

    esmlab_res = computed_dset.drop(esm.static_variables).to_dataframe()
    esmlab_res = esmlab_res.groupby('month').mean()[variables]

    df = (
        esm._ds_time_computed.drop(esm.static_variables)
        .to_dataframe()
        .reset_index(level=[esm.time_coord_name])
    )
    df[esm.time_coord_name] = df[esm.time_coord_name].values.astype('datetime64[ns]')
    pd_res = df.groupby(df[esm.time_coord_name].dt.month).mean()
    pd_res.index.name = 'month'
    pd_res = pd_res[variables]

    assert_both_frames_equal(esmlab_res, pd_res)

    assert computed_dset[esm.time_coord_name].dtype == ds[esm.time_coord_name].dtype
    for key, value in ds[esm.time_coord_name].attrs.items():
        assert key in computed_dset[esm.time_coord_name].attrs
        assert value == computed_dset[esm.time_coord_name].attrs[key]


def test_anomaly_with_monthly_clim(dset):
    computed_dset = esmlab.anomaly(dset, clim_freq='mon')
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

    computed_dset = esmlab.anomaly(
        dset, clim_freq='mon', slice_mon_clim_time=slice('0001-01-16', None)
    )
    np.testing.assert_equal(computed_dset.variable_1.values.ravel(), expected)


def test_resample_ann_mean(dset):
    weights = np.ones(24)
    computed_dset = esmlab.resample(dset, freq='ann', weights=weights)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1 / 24, 1 / 24, 1 / 24, 1 / 24], dtype=np.float32)
    np.testing.assert_allclose(
        computed_dset.variable_1.values.ravel().astype(np.float32), expected, rtol=1e-6
    )

    computed_dset = esmlab.resample(dset, freq='ann', weights=None)
    assert isinstance(computed_dset, xr.Dataset)


@pytest.mark.parametrize(
    'ds, weights',
    [('tiny', np.ones(24)), ('tiny', xr.DataArray(np.ones(24))), ('tiny', np.ones(24).tolist())],
)
def test_resample_ann_mean_values_missing(ds, weights):
    dset = open_dataset(ds, decode_times=False, decode_coords=False)
    dset.variable_1.values[0:3, :, :] = np.nan
    computed_dset = esmlab.resample(dset, freq='ann', weights=weights)
    assert isinstance(computed_dset, xr.Dataset)
    assert computed_dset.time.dtype == dset.time.dtype
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1 / 21, 1 / 21, 1 / 21, 1 / 21], dtype=np.float32)
    np.testing.assert_allclose(
        computed_dset.variable_1.values.ravel().astype(np.float32), expected, rtol=1e-6
    )


def test_resample_mon_mean(dset):
    computed_dset = esmlab.resample(dset, freq='mon')
    res = computed_dset.variable_1.data
    expected = np.full(shape=(12, 2, 2), fill_value=0.5, dtype=np.float32)
    np.testing.assert_allclose(res, expected)


def test_unsupported_args(dset):
    with pytest.raises(ValueError):
        esmlab.resample(dset, freq='hr')

    with pytest.raises(ValueError):
        esmlab.climatology(dset, freq='hr')

    with pytest.raises(ValueError):
        esmlab.anomaly(dset, clim_freq='ann')
