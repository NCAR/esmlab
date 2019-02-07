from __future__ import absolute_import, division, print_function

from datetime import datetime

import cftime
import numpy as np
import xarray as xr


def time_bound_var(ds, time_coord_name):
    tb_name = None
    tb_dim = None

    if "bounds" in ds[time_coord_name].attrs:
        tb_name = ds[time_coord_name].attrs["bounds"]
    elif "time_bound" in ds:
        tb_name = "time_bound"

    if tb_name:
        tb_dim = ds[tb_name].dims[1]
    return tb_name, tb_dim


def time_year_to_midyeardate(ds, time_coord_name):
    ds[time_coord_name].values = np.array(
        [cftime.datetime(year, 7, 2) for year in ds[time_coord_name]]
    )
    return ds


def compute_time_var(dset, tb_name, tb_dim, time_coord_name, year_offset=np.nan):

    if dset[time_coord_name].dtype != np.dtype("O"):
        time_values = dset[tb_name].mean(tb_dim)

        if not np.isnan(year_offset):
            time_values += cftime.date2num(
                datetime(int(year_offset), 1, 1),
                dset[time_coord_name].attrs["units"],
                dset[time_coord_name].attrs["calendar"],
            )
            dset[time_coord_name].attrs["esmlab_year_offset"] = year_offset

        date = cftime.num2date(
            time_values,
            units=dset[time_coord_name].attrs["units"],
            calendar=dset[time_coord_name].attrs["calendar"],
        )

        dset[time_coord_name].values = xr.CFTimeIndex(date)

    return dset


def uncompute_time_var(dset, time_coord_name):
    if dset[time_coord_name].dtype == np.dtype("O"):
        calendar = "standard"
        units = "days since 0001-01-01 00:00:00"
        if "calendar" in dset[time_coord_name].attrs:
            calendar = dset[time_coord_name].attrs["calendar"]

        if "units" in dset[time_coord_name].attrs:
            units = dset[time_coord_name].attrs["units"]

        time_values = cftime.date2num(
            dset[time_coord_name], units=units, calendar=calendar
        )

        dset[time_coord_name].values = time_values
    return dset


def infer_time_coord_name(ds):
    """ Infer name for time coordinate in a dataset"""
    if "time" in ds.variables:
        return "time"

    unlimited_dims = ds.encoding.get("unlimited_dims", None)
    if len(unlimited_dims) == 1:
        return list(unlimited_dims)[0]

    raise ValueError(
        "Cannot infer `time_coord_name` from multiple unlimited dimensions: %s \n\t\t ***** Please specify time_coord_name to use. *****"
        % unlimited_dims
    )
