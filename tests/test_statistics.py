#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import xarray as xr
import numpy as np
import pytest
from esmlab.statistics import weighted_rmsd, weighted_cov, weighted_corr

maskedarea = xr.DataArray(np.ones((10, 10)), dims=("x", "y"))
x = xr.DataArray(
    np.random.uniform(
        0, 10, (10, 10)), dims=(
            "x", "y"), name="x_var")
y = xr.DataArray(
    np.random.uniform(
        0, 10, (10, 10)), dims=(
            "x", "y"), name="y_var")
y[3, 3:10] = np.nan

test_data = [x, y]


@pytest.mark.parametrize("data", test_data)
def test_weighted_mean(data):
    np.testing.assert_allclose(
        data.esm.weighted_mean(
            weights=maskedarea),
        data.mean())


@pytest.mark.parametrize("data", test_data)
def test_weighted_std(data):
    np.testing.assert_allclose(
        data.esm.weighted_std(
            weights=maskedarea),
        data.std())


@pytest.mark.parametrize("data", test_data)
def test_weighted_sum(data):
    np.testing.assert_allclose(
        data.esm.weighted_sum(
            weights=maskedarea),
        data.sum())


def test_weighted_rmsd():
    valid = (x.notnull() & y.notnull())
    N = valid.sum()
    rmsd = np.sqrt(((x - y)**2).sum() / N)
    w_rmsd = weighted_rmsd(x, y, weights=maskedarea)
    np.testing.assert_allclose(rmsd, w_rmsd)


def test_weighted_cov():
    valid = (x.notnull() & y.notnull())
    N = valid.sum()
    x_dev = x - x.mean()
    y_dev = y - y.mean()
    cov = (x_dev * y_dev).sum() / N
    w_cov = weighted_cov(x, y, weights=maskedarea)
    np.testing.assert_allclose(cov, w_cov)


@pytest.mark.skip()
def test_weighted_corr():
    valid = (x.notnull() & y.notnull())
    N = valid.sum()
    x_dev = x - x.mean()
    y_dev = y - y.mean()
    cov = (x_dev * y_dev).sum() / N
    covx = (x_dev ** 2).sum() / N
    covy = (y_dev ** 2).sum() / N
    corr = (cov / np.sqrt(covx * covy))

    w_corr = weighted_corr(x, y, weights=maskedarea)
    np.testing.assert_allclose(corr, w_corr, rtol=1e-04)
