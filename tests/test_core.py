import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import esmlab


def test_time_bound_var(dset, time_coord_name='time'):
    results = dset.esmlab.tb_name, dset.esmlab.tb_dim
    expected = ('time_bound', 'd2')
    assert results == expected


def test_time_year_to_midyeardate(dset, time_coord_name='time'):
    assert isinstance(dset[time_coord_name].values[0], np.number)
    dset = dset.esmlab.time_year_to_midyeardate()
    assert isinstance(dset[time_coord_name].values[0], cftime.datetime)


def test_get_time_attrs(dset, time_coord_name='time'):
    expected = {
        'units': 'days since 0001-01-01 00:00:00',
        'calendar': 'noleap',
        'bounds': 'time_bound',
    }
    results = dset.esmlab.time_attrs
    assert results == expected


def test_compute_time_var(dset, time_coord_name='time'):
    idx = dset.indexes[time_coord_name]
    assert isinstance(idx, pd.core.indexes.numeric.Index)
    results = dset.esmlab.get_time_decoded()
    assert isinstance(results, xr.DataArray)


def test_uncompute_time_var(dset, time_coord_name='time'):
    ds = dset.esmlab.compute_time_var()
    assert ds[time_coord_name].dtype == np.dtype('O')
    dset_with_uncomputed_time = dset.esmlab.uncompute_time_var()
    assert np.issubdtype(dset_with_uncomputed_time[time_coord_name].dtype, np.number)


def test_sel_time(dset):
    dset = dset.esmlab.sel_time(indexer_val=slice('1850-01-01', '1850-12-31'), year_offset=1850)
    assert len(dset.time) == 12
