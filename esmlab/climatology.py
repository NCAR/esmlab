#!/usr/bin/env python
"""Contains functions to compute climatologies."""
from __future__ import absolute_import, division, print_function

import numpy as np
import xarray as xr

from .utils.common import esmlab_xr_set_options
from .utils.time import (
    compute_time_var,
    infer_time_coord_name,
    time_bound_var,
    time_year_to_midyeardate,
)
from .utils.variables import (
    get_original_attrs,
    get_static_variables,
    get_variables,
    save_metadata,
    set_metadata,
    set_static_variables,
    update_attrs,
)


@esmlab_xr_set_options(arithmetic_join="exact")
def compute_mon_climatology(dset, time_coord_name=None):
    """Calculates monthly climatology (monthly means)

    Parameters
    ----------
    dset : xarray.Dataset
           The data on which to operate

    time_coord_name : string
            Name for time coordinate

    Returns
    -------
    computed_dset : xarray.Dataset
                    The computed monthly climatology data

    """
    if time_coord_name is None:
        time_coord_name = infer_time_coord_name(dset)

    tb_name, tb_dim = time_bound_var(dset, time_coord_name)

    static_variables = get_static_variables(dset, time_coord_name)

    # save metadata
    attrs, encoding = save_metadata(dset)

    # Compute new time variable
    if tb_name and tb_dim:
        dset = compute_time_var(dset, tb_name, tb_dim, time_coord_name)

    # Compute climatology
    time_dot_month = ".".join([time_coord_name, "month"])
    computed_dset = (
        dset.drop(static_variables)
        .groupby(time_dot_month)
        .mean(time_coord_name)
        .rename({"month": time_coord_name})
    )

    # Put static_variables back
    computed_dset = set_static_variables(computed_dset, dset, static_variables)

    # add month_bounds
    computed_dset["month"] = computed_dset[time_coord_name].copy()
    attrs["month"] = {"long_name": "Month", "units": "month"}
    encoding["month"] = {"dtype": "int32", "_FillValue": None}

    if tb_name:
        computed_dset["month_bounds"] = (
            computed_dset[tb_name] - computed_dset[tb_name][0, 0]
        )
        computed_dset[time_coord_name].values = computed_dset.month_bounds.mean(
            tb_dim
        ).values

        encoding["month_bounds"] = {"dtype": "float", "_FillValue": None}
        attrs["month_bounds"] = {
            "long_name": "month_bounds",
            "units": "days since 0001-01-01 00:00:00",
        }

        attrs[time_coord_name] = {
            "long_name": time_coord_name,
            "units": "days since 0001-01-01 00:00:00",
            "bounds": "month_bounds",
        }

    if "calendar" in attrs[time_coord_name]:
        attrs[time_coord_name]["calendar"] = attrs[time_coord_name]["calendar"]
        attrs["month_bounds"]["calendar"] = attrs[time_coord_name]["calendar"]

    encoding[time_coord_name] = {"dtype": "float", "_FillValue": None}

    if tb_name:
        computed_dset = computed_dset.drop(tb_name)

    # Put the attributes, encoding back
    computed_dset = set_metadata(computed_dset, attrs, encoding, additional_attrs={})
    return computed_dset


@esmlab_xr_set_options(arithmetic_join="exact")
def compute_mon_anomaly(dset, slice_mon_clim_time=None, time_coord_name=None):
    """Calculates monthly anomaly

    Parameters
    ----------
    dset : xarray.Dataset
           The data on which to operate

    slice_mon_clim_time : slice, optional
                          a slice object passed to
                          `dset.isel(time=slice_mon_clim_time)` for subseting
                          the time-period overwhich the climatology is computed
    time_coord_name : string
            Name for time coordinate

    Returns
    -------
    computed_dset : xarray.Dataset
                    The computed monthly anomaly data

    """

    if time_coord_name is None:
        time_coord_name = infer_time_coord_name(dset)

    tb_name, tb_dim = time_bound_var(dset, time_coord_name)

    static_variables = get_static_variables(dset, time_coord_name)

    # save metadata
    attrs, encoding = save_metadata(dset)

    # Compute new time variable
    if tb_name and tb_dim:
        dset = compute_time_var(dset, tb_name, tb_dim, time_coord_name)

    # Compute anomaly
    time_dot_month = ".".join([time_coord_name, "month"])
    if slice_mon_clim_time is None:
        computed_dset = dset.drop(static_variables).groupby(time_dot_month) - dset.drop(
            static_variables
        ).groupby(time_dot_month).mean(time_coord_name)
    else:
        computed_dset = dset.drop(static_variables).groupby(time_dot_month) - dset.drop(
            static_variables
        ).sel(time=slice_mon_clim_time).groupby(time_dot_month).mean(time_coord_name)

    # reset month to become a variable
    computed_dset = computed_dset.reset_coords("month")

    # Put static_variables back
    computed_dset = set_static_variables(computed_dset, dset, static_variables)

    # Put the attributes, encoding back
    computed_dset = set_metadata(
        computed_dset,
        attrs,
        encoding,
        additional_attrs={"month": {"long_name": "Month"}},
    )
    return computed_dset


@esmlab_xr_set_options(arithmetic_join="exact")
def compute_ann_mean(dset, weights=None, time_coord_name=None):
    """Calculates annual climatology (annual means)

    Parameters
    ----------
    dset : xarray.Dataset
           The data on which to operate

    weights : array_like, optional
              weights to use for each time period.
              If None and dataset doesn't have `time_bound` variable,
              every time period has equal weight of 1.

    time_coord_name : string
            Name for time coordinate

    Returns
    -------
    computed_dset : xarray.Dataset
                    The computed annual climatology data

    """

    if time_coord_name is None:
        time_coord_name = infer_time_coord_name(dset)

    tb_name, tb_dim = time_bound_var(dset, time_coord_name)

    static_variables = get_static_variables(dset, time_coord_name)
    variables = get_variables(dset, time_coord_name, tb_name)
    # save metadata
    attrs, encoding = save_metadata(dset)

    # Compute new time variable
    if tb_name and tb_dim:
        dset = compute_time_var(dset, tb_name, tb_dim, time_coord_name)

    time_dot_year = ".".join([time_coord_name, "year"])
    # Compute weights
    if weights:
        if len(weights) != len(dset[time_coord_name]):
            raise ValueError(
                "weights and dataset time values must be of the same length"
            )
        else:
            dt = xr.ones_like(dset[time_coord_name], dtype=bool)
            dt.values = weights
            weights = dt / dt.sum(xr.ALL_DIMS)
            np.testing.assert_allclose(weights.sum(xr.ALL_DIMS), 1.0)

    elif not weights:
        if tb_name and tb_dim:

            dt = dset[tb_name].diff(dim=tb_dim)[:, 0]

            if tb_dim in dt.coords:
                dt = dt.drop(tb_dim)

            weights = dt.groupby(time_dot_year) / dt.groupby(time_dot_year).sum(
                xr.ALL_DIMS
            )

            np.testing.assert_allclose(
                weights.groupby(time_dot_year).sum(xr.ALL_DIMS), 1.0
            )

        else:
            dt = xr.ones_like(dset[time_coord_name], dtype=bool)
            weights = dt / dt.sum(xr.ALL_DIMS)
            np.testing.assert_allclose(weights.sum(xr.ALL_DIMS), 1.0)

    # groupby.sum() does not seem to handle missing values correctly: yields 0 not nan
    # the groupby.mean() does return nans, so create a mask of valid values
    # for each variable
    valid = {
        v: dset[v]
        .groupby(time_dot_year)
        .mean(dim=time_coord_name)
        .notnull()
        .rename({"year": time_coord_name})
        for v in variables
    }
    ones = (
        dset.drop(static_variables)
        .where(dset.drop(static_variables).isnull())
        .fillna(1.0)
        .where(dset.drop(static_variables).notnull())
        .fillna(0.0)
    )

    # Compute annual mean
    computed_dset = (
        (dset.drop(static_variables) * weights)
        .groupby(time_dot_year)
        .sum(time_coord_name)
        .rename({"year": time_coord_name})
    )
    ones_out = (
        (ones * weights)
        .groupby(time_dot_year)
        .sum(time_coord_name)
        .rename({"year": time_coord_name})
    )
    ones_out = ones_out.where(ones_out > 0.0)

    # Renormalize to appropriately account for missing values
    computed_dset = computed_dset / ones_out

    # Apply the valid-values mask
    for v in variables:
        computed_dset[v] = computed_dset[v].where(valid[v])

    # address time
    attrs[time_coord_name] = {
        "long_name": time_coord_name,
        "units": "days since 0001-01-01 00:00:00",
    }

    if "calendar" in attrs[time_coord_name]:
        attrs[time_coord_name]["calendar"] = attrs[time_coord_name]["calendar"]

    # compute the time_bound variable
    if tb_name and tb_dim:
        tb_out_lo = (
            dset[tb_name][:, 0]
            .groupby(time_dot_year)
            .min(dim=time_coord_name)
            .rename({"year": time_coord_name})
        )
        tb_out_hi = (
            dset[tb_name][:, 1]
            .groupby(time_dot_year)
            .max(dim=time_coord_name)
            .rename({"year": time_coord_name})
        )

        computed_dset[tb_name] = xr.concat((tb_out_lo, tb_out_hi), dim=tb_dim)
        attrs[time_coord_name]["bounds"] = tb_name

    # Put static_variables back
    computed_dset = set_static_variables(computed_dset, dset, static_variables)

    # make year into date
    computed_dset = time_year_to_midyeardate(computed_dset, time_coord_name)

    # Put the attributes, encoding back
    computed_dset = set_metadata(computed_dset, attrs, encoding, additional_attrs={})

    return computed_dset
