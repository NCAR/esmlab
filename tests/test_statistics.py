#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import sys

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal

from esmlab import statistics

INPUT = xr.DataArray(
    [[2, 20, 25], [3, 30, 35], [4, 40, 45], [5, 50, 55]],
    dims=["t", "s"],
    coords={
        "t": np.array(
            ["1990-12-30", "2000-12-30", "2005-12-30", "2010-12-30"], dtype="<M8[D]"
        ),
        "s": ["s1", "s2", "s3"],
    },
)


T_COMPOUND_MATRIX = xr.DataArray(
    np.array(
        [
            ["1990-12-30", "NaT", "NaT"],
            ["1990-12-30", "2005-12-30", "NaT"],
            ["2000-12-30", "1990-12-30", "NaT"],
            ["2010-12-30", "1990-12-30", "2005-12-30"],
        ],
        dtype="<M8[D]",
    ),
    dims=["t2", "c"],
    coords={"t2": [10, 20, 30, 40]},
)


S_COMPOUND_MATRIX = xr.DataArray(
    [["s3", "s2"], ["s1", ""]], dims=["s2", "c"], coords={"s2": ["foo", "bar"]}
)

DTYPES = ("int64", "float32", "float64")


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


@pytest.mark.skipif(
    sys.version_info < (3, 5),
    reason="xarray-extras does not support Python version < 3.5",
)
@pytest.mark.parametrize("dtype", [float, int, "complex128"])
@pytest.mark.parametrize("skipna", [False, True, None])
@pytest.mark.parametrize("use_dask", [False, True])
def test_cummean(use_dask, skipna, dtype):
    x = INPUT.copy(deep=True).astype(dtype)
    if dtype in (float, "complex128"):
        x[2, 1] = np.nan

    expect = xr.concat(
        [
            x[:1].mean("t", skipna=skipna),
            x[:2].mean("t", skipna=skipna),
            x[:3].mean("t", skipna=skipna),
            x[:4].mean("t", skipna=skipna),
        ],
        dim="t",
    )
    expect.coords["t"] = x.coords["t"]
    if use_dask:
        x = x.chunk({"s": 2, "t": 3})
        expect = expect.chunk({"s": 2, "t": 3})

    actual = statistics.cummean(x, "t", skipna=skipna)
    assert_equal(expect, actual)
    assert expect.dtype == actual.dtype


@pytest.mark.parametrize(
    "func, meth",
    [
        (statistics.compound_sum, "sum"),
        (statistics.compound_prod, "prod"),
        (statistics.compound_mean, "mean"),
    ],
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("use_dask", [False, True])
def test_compound_t(func, meth, dtype, use_dask):
    x = INPUT.astype(dtype)
    c = T_COMPOUND_MATRIX
    expect = xr.concat(
        [
            getattr(x.isel(t=[0]), meth)("t"),
            getattr(x.isel(t=[0, 2]), meth)("t"),
            getattr(x.isel(t=[1, 0]), meth)("t"),
            getattr(x.isel(t=[3, 0, 2]), meth)("t"),
        ],
        dim="t2",
    ).T.astype(dtype)
    expect.coords["t2"] = c.coords["t2"]

    if use_dask:
        x = x.chunk({"s": 2})
        expect = expect.chunk({"s": 2})
        c = c.chunk()

    actual = func(x, c, "t", "c")

    if use_dask:
        assert_equal(expect.compute(), actual.compute())
    else:
        assert_equal(expect, actual)

    assert expect.dtype == actual.dtype
    assert actual.chunks == expect.chunks


@pytest.mark.parametrize(
    "func, meth",
    [
        (statistics.compound_sum, "sum"),
        (statistics.compound_prod, "prod"),
        (statistics.compound_mean, "mean"),
    ],
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("use_dask", [False, True])
def test_compound_s(func, meth, dtype, use_dask):
    x = INPUT.astype(dtype)
    c = S_COMPOUND_MATRIX
    expect = xr.concat(
        [
            getattr(x.sel(s=["s3", "s2"]), meth)("s"),
            getattr(x.sel(s=["s1"]), meth)("s"),
        ],
        dim="s2",
    ).T.astype(dtype)
    expect.coords["s2"] = c.coords["s2"]

    if use_dask:
        x = x.chunk({"t": 2})
        expect = expect.chunk({"t": 2})
        c = c.chunk()

    actual = func(x, c, "s", "c")

    if use_dask:
        assert_equal(expect.compute(), actual.compute())
    else:
        assert_equal(expect, actual)

    assert expect.dtype == actual.dtype
    assert actual.chunks == expect.chunks
