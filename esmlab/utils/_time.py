from __future__ import absolute_import, division, print_function

from datetime import datetime

import cftime
import numpy as np
import xarray as xr


def time_bound_var(ds):
    tb_name = None
    tb_dim = None

    if "bounds" in ds["time"].attrs:
        tb_name = ds["time"].attrs["bounds"]
    elif "time_bound" in ds:
        tb_name = "time_bound"

    if tb_name:
        tb_dim = ds[tb_name].dims[1]
    return tb_name, tb_dim


def time_year_to_midyeardate(ds):
    ds.time.values = np.array([cftime.datetime(year, 7, 2) for year in ds.time])
    return ds


def compute_time_var(dset, tb_name, tb_dim, year_offset=np.nan):

    if dset.time.dtype != np.dtype("O"):
        time_values = dset[tb_name].mean(tb_dim)

        if not np.isnan(year_offset):
            time_values += cftime.date2num(
                datetime(int(year_offset), 1, 1),
                dset.time.attrs["units"],
                dset.time.attrs["calendar"],
            )
            dset.time.attrs["esmlab_year_offset"] = year_offset

        date = cftime.num2date(
            time_values,
            units=dset.time.attrs["units"],
            calendar=dset.time.attrs["calendar"],
        )

        dset.time.values = xr.CFTimeIndex(date)

    return dset


def uncompute_time_var(dset):
    if dset.time.dtype == np.dtype("O"):
        calendar = "standard"
        units = "days since 0001-01-01 00:00:00"
        if "calendar" in dset.time.attrs:
            calendar = dset.time.attrs["calendar"]

        if "units" in dset.time.attrs:
            units = dset.time.attrs["units"]

        time_values = cftime.date2num(dset.time, units=units, calendar=calendar)

        dset.time.values = time_values
    return dset
