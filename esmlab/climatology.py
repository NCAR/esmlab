#!/usr/bin/env python
"""Contains functions to compute climatologies."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import xarray as xr
from .utils import (
    time_bound_var,
    get_grid_vars,
    save_metadata,
    compute_time_var,
    set_grid_vars,
    set_metadata,
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
    computed_dset = set_metadata(computed_dset, attrs, encoding, additional_attrs={
                                 'time': {"long_name": "Month", "units": "month"}})
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
    computed_dset = (
        dset.drop(grid_vars)
        .groupby("time.month") - dset.drop(grid_vars).groupby('time.month').mean("time")
    )

    # reset month to become a variable
    computed_dset = computed_dset.reset_coords('month')

    # Put grid_vars back
    computed_dset = set_grid_vars(computed_dset, dset, grid_vars)

    # Put the attributes, encoding back
    computed_dset = set_metadata(
        computed_dset,
        attrs,
        encoding,
        additional_attrs={
            "month": {
                'long_name': 'Month'}})
    return computed_dset
