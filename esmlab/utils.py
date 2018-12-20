from __future__ import absolute_import, division, print_function

import cftime
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
            dset[v].encoding = encoding[v]

    return dset


def compute_time_var(dset, tb_name, tb_dim):
    # -- compute time variable
    date = cftime.num2date(
        dset[tb_name].mean(tb_dim),
        units=dset.time.attrs["units"],
        calendar=dset.time.attrs["calendar"],
    )
    if len(date) % 12 != 0:
        raise ValueError("Time axis not evenly divisible by 12!")

    else:
        dset.time.values = date

    return dset


def set_grid_vars(computed_dset, dset, grid_vars):
    return xr.merge(
        (computed_dset, dset.drop([v for v in dset.variables if v not in grid_vars]))
    )


def get_original_attrs(x):
    attrs = x.attrs.copy()
    encoding = x.encoding
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
