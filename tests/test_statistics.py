#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import xarray as xr

from esmlab import statistics

x = xr.DataArray(np.random.uniform(0, 10, (10, 10)), dims=('x', 'y'), name='x_var')
y = xr.DataArray(np.random.uniform(0, 10, (10, 10)), dims=('x', 'y'), name='y_var')
y[3, 3:10] = np.nan
valid = x.notnull() & y.notnull()
N = valid.sum()
x = x.where(valid)
y = y.where(valid)

data1 = np.arange(27, dtype='float32').reshape(3, 3, 3)
data2 = np.arange(9, dtype='float32').reshape(3, 3)
data2[1, 2:3] = np.nan
data1[0, 2] = np.nan
data1[2, 1:3] = np.nan
da1 = xr.DataArray(data1, dims=['x', 'y', 'z'])
da2 = xr.DataArray(data2, dims=['x', 'y'])
ds = xr.Dataset({'variable_x': (['x', 'y', 'z'], da1), 'variable_y': (['x', 'y'], da2)})


@pytest.mark.parametrize('data', [da1, da2])
def test_weighted_mean_da(data):
    with pytest.warns(UserWarning):
        w_mean = statistics.weighted_mean(data)
    np.testing.assert_allclose(w_mean, data.mean())
    assert data.attrs == w_mean.attrs
    assert data.encoding == w_mean.encoding


@pytest.mark.parametrize('data', [da1, da2])
def test_weighted_mean_da_with_dim(data, dim=['x', 'y']):
    with pytest.warns(UserWarning):
        w_mean = statistics.weighted_mean(data, dim)
    np.testing.assert_allclose(w_mean, data.mean(dim))
    assert data.attrs == w_mean.attrs
    assert data.encoding == w_mean.encoding


def test_weighted_mean_ds():
    with pytest.warns(UserWarning):
        w_mean = statistics.weighted_mean(ds)
    np.testing.assert_allclose(w_mean['variable_x'], ds['variable_x'].mean())


@pytest.mark.skip
def test_weighted_std():
    with pytest.warns(UserWarning):
        w_std = statistics.weighted_std(x)
    np.testing.assert_allclose(w_std, x.std())
    assert x.attrs == w_std.attrs
    assert x.encoding == w_std.encoding


@pytest.mark.skip
def test_weighted_sum():
    with pytest.warns(UserWarning):
        w_sum = statistics.weighted_sum(x)
    np.testing.assert_allclose(w_sum, x.sum())
    assert x.attrs == w_sum.attrs
    assert x.encoding == w_sum.encoding


@pytest.mark.skip
def test_weighted_rmsd():
    rmsd = np.sqrt(((x - y) ** 2).sum() / N)
    with pytest.warns(UserWarning):
        w_rmsd = statistics.weighted_rmsd(x, y)
        np.testing.assert_allclose(rmsd, w_rmsd)


@pytest.mark.skip
def test_weighted_cov():
    x_dev = x - x.mean()
    y_dev = y - y.mean()
    cov = (x_dev * y_dev).sum() / N
    with pytest.warns(UserWarning):
        w_cov = statistics.weighted_cov(x, y)
        np.testing.assert_allclose(cov, w_cov)


@pytest.mark.skip
def test_weighted_corr():
    x_dev = x - x.mean()
    y_dev = y - y.mean()
    cov = (x_dev * y_dev).sum() / N
    covx = (x_dev ** 2).sum() / N
    covy = (y_dev ** 2).sum() / N
    corr = cov / np.sqrt(covx * covy)

    with pytest.warns(UserWarning):
        w_corr = statistics.weighted_corr(x, y)
        np.testing.assert_allclose(corr, w_corr)


@pytest.mark.skip
def test_weighted_sum_float32():
    from esmlab.datasets import open_dataset

    with open_dataset('cesm_pop_monthly') as ds:
        weights = ds['TAREA'].astype(np.float32)
        tmp_data = ds['TLAT']
        tmp_data.values = np.where(np.greater_equal(ds['KMT'].values, 1), ds['TLAT'], np.nan)
        tmp_data = tmp_data.astype(np.float32)
    w_sum = statistics.weighted_sum(tmp_data, weights)
    assert tmp_data.attrs == w_sum.attrs
    assert tmp_data.encoding == w_sum.encoding
