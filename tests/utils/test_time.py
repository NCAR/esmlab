import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from esmlab import utils


def test_time_bound_var(dset, time_coord_name="time"):
    results = utils.time.time_bound_var(dset, time_coord_name)
    expected = ("time_bound", "d2")
    assert results == expected


def test_time_year_to_midyeardate(dset, time_coord_name="time"):
    assert isinstance(dset[time_coord_name].values[0], np.number)
    dset = utils.time.time_year_to_midyeardate(dset, time_coord_name)
    assert isinstance(dset[time_coord_name].values[0], cftime.datetime)


def test_get_time_attrs(dset, time_coord_name="time"):
    expected = {"units": "days since 0001-01-01 00:00:00", "calendar": "noleap"}
    results = utils.time.get_time_attrs(dset, time_coord_name)
    assert results == expected


def test_compute_time_var(dset, time_coord_name="time"):
    idx = dset.indexes[time_coord_name]
    assert isinstance(idx, pd.core.indexes.numeric.Index)
    tb_name, tb_dim = utils.time.time_bound_var(dset, time_coord_name)
    results = utils.time.compute_time_var(
        dset, tb_name, tb_dim, time_coord_name
    ).indexes[time_coord_name]
    assert isinstance(results, xr.coding.cftimeindex.CFTimeIndex)


def test_uncompute_time_var(dset, time_coord_name="time"):
    tb_name, tb_dim = utils.time.time_bound_var(dset, time_coord_name)
    dset = utils.time.compute_time_var(dset, tb_name, tb_dim, time_coord_name)
    assert dset[time_coord_name].dtype == np.dtype("O")
    dset_with_uncomputed_time = utils.time.uncompute_time_var(dset, time_coord_name)
    assert np.issubdtype(dset_with_uncomputed_time[time_coord_name].dtype, np.number)
