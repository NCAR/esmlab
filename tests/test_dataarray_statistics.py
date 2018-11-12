#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import xarray as xr
import numpy as np
import pytest
import esmlab

esmlab.__version__

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
