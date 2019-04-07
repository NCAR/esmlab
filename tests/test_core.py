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
    esm = ds.esmlab
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


@pytest.mark.parametrize(
    'ds_name, decoded, variables',
    [
        ('tiny', True, ['variable_1', 'variable_2']),
        ('tiny', False, ['variable_1', 'variable_2']),
        ('cmip5_pr_amon_csiro', False, ['pr']),
        ('cmip5_pr_amon_csiro', True, ['pr']),
        ('cesm_cice_daily', False, ['aicen_d']),
        ('cesm_pop_daily', True, ['FG_CO2_2']),
    ],
)
def test_mon_climatology(ds_name, decoded, variables):
    ds = esmlab.datasets.open_dataset(ds_name, decode_times=decoded)
    esm = ds.esmlab
    esmlab_res = esmlab.climatology(ds, freq='mon').drop(esm.static_variables).to_dataframe()
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
