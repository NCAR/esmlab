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
    time_year_to_midyeardate,
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

    # add month_bounds
    computed_dset["month"] = computed_dset.time.copy()
    attrs["month"] = {"long_name": "Month", "units": "month"}
    encoding["month"] = {"dtype": "int32", "_FillValue": None}

    if tb_name:
        computed_dset["month_bounds"] = (
            computed_dset[tb_name] - computed_dset[tb_name][0, 0]
        )
        computed_dset.time.values = computed_dset.month_bounds.mean(tb_dim).values

        encoding["month_bounds"] = {"dtype": "float", "_FillValue": None}
        attrs["month_bounds"] = {
            "long_name": "month_bounds",
            "units": "days since 0001-01-01 00:00:00",
        }

        attrs["time"] = {
            "long_name": "time",
            "units": "days since 0001-01-01 00:00:00",
            "bounds": "month_bounds",
        }

    if "calendar" in attrs["time"]:
        attrs["time"]["calendar"] = attrs["time"]["calendar"]
        attrs["month_bounds"]["calendar"] = attrs["time"]["calendar"]

    encoding["time"] = {"dtype": "float", "_FillValue": None}

    if tb_name:
        computed_dset = computed_dset.drop(tb_name)

    # Put the attributes, encoding back
    computed_dset = set_metadata(computed_dset, attrs, encoding, additional_attrs={})
    return computed_dset


def compute_mon_anomaly(dset, slice_mon_clim_time=None):
    """Calculates monthly anomaly

    Parameters
    ----------
    dset : xarray.Dataset
           The data on which to operate

    slice_mon_clim_time : slice, optional
                          a slice object passed to
                          `dset.isel(time=slice_mon_clim_time)` for subseting
                          the time-period overwhich the climatology is computed

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
    if slice_mon_clim_time is None:
        computed_dset = dset.drop(grid_vars).groupby("time.month") - dset.drop(
            grid_vars
        ).groupby("time.month").mean("time")
    else:
        computed_dset = dset.drop(grid_vars).groupby("time.month") - dset.drop(
            grid_vars
        ).sel(time=slice_mon_clim_time).groupby("time.month").mean("time")

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


def compute_ann_mean(dset, weights=None):
    """Calculates annual climatology (annual means)

    Parameters
    ----------
    dset : xarray.Dataset
           The data on which to operate

    weights : array_like, optional
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
            weights = dt / dt.sum(xr.ALL_DIMS)
            np.testing.assert_allclose(weights.sum(xr.ALL_DIMS), 1.0)

    elif not weights:
        if tb_name and tb_dim:

            dt = dset[tb_name].diff(dim=tb_dim)[:, 0]

            if tb_dim in dt.coords:
                dt = dt.drop(tb_dim)

            weights = dt.groupby("time.year") / dt.groupby("time.year").sum(xr.ALL_DIMS)

            np.testing.assert_allclose(
                weights.groupby("time.year").sum(xr.ALL_DIMS), 1.0
            )

        else:
            dt = xr.ones_like(dset.time, dtype=bool)
            weights = dt / dt.sum(xr.ALL_DIMS)
            np.testing.assert_allclose(weights.sum(xr.ALL_DIMS), 1.0)

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

    # Compute annual mean
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

    if tb_name:
        computed_dset = computed_dset.drop(tb_name)

    if tb_dim in computed_dset.dims:
        computed_dset = computed_dset.drop(tb_dim)

    # Apply the valid-values mask
    for v in variables:
        computed_dset[v] = computed_dset[v].where(valid[v])

    # Put grid_vars back
    computed_dset = set_grid_vars(computed_dset, dset, grid_vars)

    # make year into date
    computed_dset = time_year_to_midyeardate(computed_dset)

    attrs["time"] = {"long_name": "time", "units": "days since 0001-01-01 00:00:00"}

    if "calendar" in attrs["time"]:
        attrs["time"]["calendar"] = attrs["time"]["calendar"]
        attrs["month_bounds"]["calendar"] = attrs["time"]["calendar"]

    # Put the attributes, encoding back
    computed_dset = set_metadata(computed_dset, attrs, encoding, additional_attrs={})

    return computed_dset
