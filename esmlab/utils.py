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


def get_grid_vars(dset):
    return [v for v in dset.variables if "time" not in dset[v].dims]


def get_variables(dset, tb_name=None):
    return [
        v
        for v in dset.variables
        if "time" in dset[v].dims and v not in ["time", tb_name]
    ]


def save_metadata(dset):
    attrs = {v: dset[v].attrs for v in dset.variables}
    encoding = {
        v: {
            key: val
            for key, val in dset[v].encoding.items()
            if key in ["dtype", "_FillValue", "missing_value"]
        }
        for v in dset.variables
    }
    return attrs, encoding


def set_metadata(dset, attrs, encoding, additional_attrs):
    attrs.update(additional_attrs)

    # put the attributes back
    for v in dset.variables:
        dset[v].attrs = attrs[v]

    # put the encoding back
    for v in dset.variables:
        if v in encoding:
            if dset[v].dtype == "int64":  # int64 breaks some other tools
                encoding[v]["dtype"] = "int32"
            dset[v].encoding = encoding[v]

    return dset


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


def set_grid_vars(computed_dset, dset, grid_vars):
    return xr.merge(
        (computed_dset, dset.drop([v for v in dset.variables if v not in grid_vars]))
    )


def get_original_attrs(x):
    attrs = x.attrs.copy()
    encoding = x.encoding
    if "_FillValue" not in encoding:
        encoding["_FillValue"] = None
    return attrs, encoding


def update_attrs(x, original_attrs, original_encoding):
    for att in ["grid_loc", "coordinates"]:
        if att in original_attrs:
            del original_attrs[att]

    x.attrs = original_attrs
    x.encoding = {
        key: val
        for key, val in original_encoding.items()
        if key in ["_FillValue", "dtype"]
    }
    return x
