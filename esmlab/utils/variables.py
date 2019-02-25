from __future__ import absolute_import, division, print_function

from datetime import datetime

import cftime
import numpy as np
import xarray as xr


def get_static_variables(dset, time_coord_name):
    return [v for v in dset.variables if time_coord_name not in dset[v].dims]


def get_variables(dset, time_coord_name, tb_name=None):
    return [
        v
        for v in dset.variables
        if time_coord_name in dset[v].dims and v not in [time_coord_name, tb_name]
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


def set_static_variables(computed_dset, dset, static_variables):
    return xr.merge(
        (computed_dset, dset.drop([v for v in dset.variables if v not in static_variables]))
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
        key: val for key, val in original_encoding.items() if key in ["_FillValue", "dtype"]
    }
    return x
