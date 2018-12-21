#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import xarray as xr

from esmlab import statistics


def _generate_data(raw):

    maskedarea = xr.DataArray(np.ones((10, 10)), dims=("x", "y"))
    x = xr.DataArray(np.random.uniform(0, 10, (10, 10)), dims=("x", "y"), name="x_var")
    y = xr.DataArray(np.random.uniform(0, 10, (10, 10)), dims=("x", "y"), name="y_var")
    y[3, 3:10] = np.nan

    valid = x.notnull() & y.notnull()
    N = valid.sum()
    x = x.where(valid)
    y = y.where(valid)

    if raw:
        return [(x, y, N, maskedarea)]

    else:
        test_data = [(x, maskedarea), (y, maskedarea)]
        return test_data


def generate_data_1():
    return _generate_data(raw=True)


def generate_data_2():
    return _generate_data(raw=False)


@pytest.mark.parametrize("data, maskedarea", generate_data_2())
def test_weighted_mean(data, maskedarea):
    w_mean = statistics.weighted_mean(data, weights=maskedarea)
    np.testing.assert_allclose(w_mean, data.mean())
    assert data.attrs == w_mean.attrs
    assert data.encoding == w_mean.encoding


@pytest.mark.parametrize("data, maskedarea", generate_data_2())
def test_weighted_std(data, maskedarea):
    w_std = statistics.weighted_std(data, weights=maskedarea)
    np.testing.assert_allclose(w_std, data.std())
    assert data.attrs == w_std.attrs
    assert data.encoding == w_std.encoding


@pytest.mark.parametrize("data, maskedarea", generate_data_2())
def test_weighted_sum(data, maskedarea):
    w_sum = statistics.weighted_sum(data, weights=maskedarea)
    np.testing.assert_allclose(w_sum, data.sum())
    assert data.attrs == w_sum.attrs
    assert data.encoding == w_sum.encoding


@pytest.mark.parametrize("x, y, N, maskedarea", generate_data_1())
def test_weighted_rmsd(x, y, N, maskedarea):
    rmsd = np.sqrt(((x - y) ** 2).sum() / N)
    w_rmsd = statistics.weighted_rmsd(x, y, weights=maskedarea)
    np.testing.assert_allclose(rmsd, w_rmsd)


@pytest.mark.parametrize("x, y, N, maskedarea", generate_data_1())
def test_weighted_cov(x, y, N, maskedarea):
    x_dev = x - x.mean()
    y_dev = y - y.mean()
    cov = (x_dev * y_dev).sum() / N
    w_cov = statistics.weighted_cov(x, y, weights=maskedarea)
    np.testing.assert_allclose(cov, w_cov)


@pytest.mark.parametrize("x, y, N, maskedarea", generate_data_1())
def test_weighted_corr(x, y, N, maskedarea):
    x_dev = x - x.mean()
    y_dev = y - y.mean()
    cov = (x_dev * y_dev).sum() / N
    covx = (x_dev ** 2).sum() / N
    covy = (y_dev ** 2).sum() / N
    corr = cov / np.sqrt(covx * covy)

    w_corr = statistics.weighted_corr(x, y, weights=maskedarea)
    np.testing.assert_allclose(corr, w_corr)
