#!/usr/bin/env python
"""Contains functions to compute climatologies."""
from __future__ import absolute_import, division, print_function

import numpy as np
import xarray as xr

from .utils import (
    compute_time_var,
    get_grid_vars,
    get_variables,
    save_metadata,
    set_grid_vars,
    set_metadata,
    time_bound_var,
)


def compute_mon_climatology(dset):
    """Calculates monthly climatology (monthly means)

    Parameters
    ----------
    dset : xarray.Dataset
           The data on which to operate

    Returns
    -------
    computed_dset : xarray.Dataset
                    The computed monthly climatology data

    """

    tb_name, tb_dim = time_bound_var(dset)

    grid_vars = get_grid_vars(dset)

    # save metadata
    attrs, encoding = save_metadata(dset)

    # Compute new time variable
    if tb_name and tb_dim:
        dset = compute_time_var(dset, tb_name, tb_dim)

    # Compute climatology
    computed_dset = (
        dset.drop(grid_vars)
        .groupby("time.month")
        .mean("time")
        .rename({"month": "time"})
    )

    # Put grid_vars back
    computed_dset = set_grid_vars(computed_dset, dset, grid_vars)

    # Put the attributes, encoding back
    computed_dset = set_metadata(
        computed_dset,
        attrs,
        encoding,
        additional_attrs={"time": {"long_name": "Month", "units": "month"}},
    )
    return computed_dset


def compute_mon_anomaly(dset):
    """Calculates monthly anomaly

    Parameters
    ----------
    dset : xarray.Dataset
           The data on which to operate

    Returns
    -------
    computed_dset : xarray.Dataset
                    The computed monthly anomaly data

    """

    tb_name, tb_dim = time_bound_var(dset)

    grid_vars = get_grid_vars(dset)

    # save metadata
    attrs, encoding = save_metadata(dset)

    # Compute new time variable
    if tb_name and tb_dim:
        dset = compute_time_var(dset, tb_name, tb_dim)

    # Compute anomaly
    computed_dset = dset.drop(grid_vars).groupby("time.month") - dset.drop(
        grid_vars
    ).groupby("time.month").mean("time")

    # reset month to become a variable
    computed_dset = computed_dset.reset_coords("month")

    # Put grid_vars back
    computed_dset = set_grid_vars(computed_dset, dset, grid_vars)

    # Put the attributes, encoding back
    computed_dset = set_metadata(
        computed_dset,
        attrs,
        encoding,
        additional_attrs={"month": {"long_name": "Month"}},
    )
    return computed_dset


def compute_ann_climatology(dset, weights=None):
    """Calculates annual climatology (annual means)

    Parameters
    ----------
    dset : xarray.Dataset
           The data on which to operate

    weights : array_like
              weights to use for each time period.
              If None and dataset doesn't have `time_bound` variable,
              every time period has equal weight of 1.

    Returns
    -------
    computed_dset : xarray.Dataset
                    The computed annual climatology data

    """

    tb_name, tb_dim = time_bound_var(dset)

    grid_vars = get_grid_vars(dset)
    variables = get_variables(dset, tb_name)
    # save metadata
    attrs, encoding = save_metadata(dset)

    # Compute new time variable
    if tb_name and tb_dim:
        dset = compute_time_var(dset, tb_name, tb_dim)

    # Compute weights
    if weights:
        if len(weights) != len(dset.time):
            raise ValueError(
                "weights and dataset time values must be of the same length"
            )
        else:
            dt = xr.ones_like(dset.time, dtype=bool)
            dt.values = weights
            weights = dt / dt.sum(dim=xr.ALL_DIMS)
            np.testing.assert_allclose(weights.sum(dim=xr.ALL_DIMS), 1.0)

    elif not weights:
        if tb_name and tb_dim:
            dt = dset[tb_name].diff(dim=tb_dim)[:, 0]
            weights = dt.groupby("time.year") / dt.groupby("time.year").sum(
                dim=xr.ALL_DIMS
            )
            np.testing.assert_allclose(
                weights.groupby("time.year").sum(dim=xr.ALL_DIMS), 1.0
            )

        else:
            dt = xr.ones_like(dset.time, dtype=bool)
            weights = dt / dt.sum(dim=xr.ALL_DIMS)
            np.testing.assert_allclose(weights.sum(dim=xr.ALL_DIMS), 1.0)

    # groupby.sum() does not seem to handle missing values correctly: yields 0 not nan
    # the groupby.mean() does return nans, so create a mask of valid values
    # for each variable
    valid = {
        v: dset[v]
        .groupby("time.year")
        .mean(dim="time")
        .notnull()
        .rename({"year": "time"})
        for v in variables
    }
    ones = (
        dset.drop(grid_vars)
        .where(dset.isnull())
        .fillna(1.0)
        .where(dset.notnull())
        .fillna(0.0)
    )

    # Compute annual means
    computed_dset = (
        (dset.drop(grid_vars) * weights)
        .groupby("time.year")
        .sum("time")
        .rename({"year": "time"})
    )
    ones_out = (
        (ones * weights).groupby("time.year").sum("time").rename({"year": "time"})
    )
    ones_out = ones_out.where(ones_out > 0.0)

    # Renormalize to appropriately account for missing values
    computed_dset = computed_dset / ones_out

    # Put grid_vars back
    computed_dset = set_grid_vars(computed_dset, dset, grid_vars)

    # Apply the valid-values mask
    for v in variables:
        computed_dset[v] = computed_dset[v].where(valid[v])

    # Put the attributes, encoding back
    computed_dset = set_metadata(computed_dset, attrs, encoding, additional_attrs={})
    return computed_dset
