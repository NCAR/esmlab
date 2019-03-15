#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import xarray as xr

from esmlab import statistics

data1 = np.arange(27, dtype='float32').reshape(3, 3, 3)
data2 = np.arange(9, dtype='float32').reshape(3, 3)
data2[1, 2:3] = np.nan
data1[0, 2] = np.nan
data1[2, 1:3] = np.nan
da1 = xr.DataArray(data1, dims=['x', 'y', 'z'])
da2 = xr.DataArray(data2, dims=['x', 'y'])
da3 = xr.DataArray(np.random.rand(3, 3, 3, 3).astype('float32'), dims=['time', 'x', 'y', 'z'])
ds = xr.Dataset({'variable_x': (['x', 'y', 'z'], da1), 'variable_y': (['x', 'y'], da2)})
weights1 = xr.DataArray(
    np.arange(27, 0, -1, dtype='float32').reshape(3, 3, 3), dims=['x', 'y', 'z']
)
weights2 = xr.DataArray(
    np.arange(81, 0, -1, dtype='float32').reshape(3, 3, 3, 3), dims=['time', 'x', 'y', 'z']
)
test_data_da = [
    (da1, ['x', 'y', 'z'], weights1, (0, 1, 2)),
    (da1, None, None, None),
    (da2, ['x', 'y'], None, (0, 1)),
    (da3, ['time', 'z'], weights2, (0, 3)),
]

test_data_ds = [(ds, ['x', 'y']), (ds, None)]


@pytest.mark.parametrize('data,dim,weights,axis', test_data_da)
def test_weighted_sum_da(data, dim, weights, axis):
    with pytest.warns(UserWarning):
        w_sum = statistics.weighted_sum(data, dim)
    np.testing.assert_allclose(w_sum, data.sum(dim))
    assert data.attrs == w_sum.attrs
    assert data.encoding == w_sum.encoding


@pytest.mark.parametrize('data,dim', test_data_ds)
def test_weighted_sum_ds(data, dim):
    with pytest.warns(UserWarning):
        w_sum = statistics.weighted_sum(ds, dim)
    np.testing.assert_allclose(w_sum['variable_x'], ds['variable_x'].sum(dim))


@pytest.mark.parametrize('data,dim,weights,axis', test_data_da)
def test_weighted_mean_da(data, dim, weights, axis):
    ma = np.ma.MaskedArray(data, mask=np.isnan(data))
    np_w_mean = np.ma.average(ma, axis=axis, weights=weights)

    w_mean = statistics.weighted_mean(data, dim, weights)

    np.testing.assert_allclose(w_mean, np_w_mean)
    assert data.attrs == w_mean.attrs
    assert data.encoding == w_mean.encoding


@pytest.mark.parametrize('data,dim', test_data_ds)
def test_weighted_mean_ds(data, dim):
    with pytest.warns(UserWarning):
        w_mean = statistics.weighted_mean(ds, dim)
    np.testing.assert_allclose(w_mean['variable_x'], ds['variable_x'].mean(dim))


@pytest.mark.parametrize('data,dim,weights,axis', test_data_da)
def test_weighted_std_da(data, dim, weights, axis):
    w_std = statistics.weighted_std(data, dim)
    np.testing.assert_allclose(w_std, data.std(dim))
    assert data.attrs == w_std.attrs
    assert data.encoding == w_std.encoding


@pytest.mark.parametrize('data,dim', test_data_ds)
def test_weighted_std_ds(data, dim):
    w_std = statistics.weighted_std(ds, dim)
    np.testing.assert_allclose(w_std['variable_x'], ds['variable_x'].std(dim))


def test_weighted_rmsd_da():
    dim = ['x', 'y']
    valid = da1.notnull() & da2.notnull()
    N = valid.sum(dim)
    rmsd = np.sqrt(((da1 - da2) ** 2).sum(dim) / N)
    w_rmsd = statistics.weighted_rmsd(da1, da2, dim)
    np.testing.assert_allclose(rmsd, w_rmsd)


def test_weighted_rmsd_ds():
    rmsd = statistics.weighted_rmsd(ds, ds).to_array().values
    np.testing.assert_allclose(rmsd, np.array([0.0, 0.0]))


def test_weighted_cov():
    dim = ['x', 'y']
    valid = da1.notnull() & da2.notnull()
    N = valid.sum(dim)
    da1_dev = da1 - da1.mean(dim)
    da2_dev = da2 - da2.mean(dim)
    cov = (da1_dev * da2_dev).sum(dim) / N

    w_cov = statistics.weighted_cov(da1, da2, dim)
    np.testing.assert_allclose(cov, w_cov)


def test_weighted_corr():
    dim = ['x', 'y']
    valid = da1.notnull() & da2.notnull()
    da1_ = da1.where(valid)
    da2_ = da2.where(valid)
    N = valid.sum(dim)
    da1_dev = da1_ - da1_.mean(dim)
    da2_dev = da2_ - da2_.mean(dim)
    cov = (da1_dev * da2_dev).sum(dim) / N
    covx = (da1_dev ** 2).sum(dim) / N
    covy = (da2_dev ** 2).sum(dim) / N
    corr = cov / np.sqrt(covx * covy)

    w_corr = statistics.weighted_corr(da1, da2, dim)
    np.testing.assert_allclose(corr, w_corr)


def test_weighted_sum_float32():
    from esmlab.datasets import open_dataset

    with open_dataset('cesm_pop_monthly') as ds:
        weights = ds['TAREA'].astype(np.float32)
        tmp_data = ds['TLAT']
        tmp_data.values = np.where(np.greater_equal(ds['KMT'].values, 1), ds['TLAT'], np.nan)
        tmp_data = tmp_data.astype(np.float32)
    w_sum = statistics.weighted_sum(tmp_data, dim=None, weights=weights)
    assert tmp_data.attrs == w_sum.attrs
    assert tmp_data.encoding == w_sum.encoding
