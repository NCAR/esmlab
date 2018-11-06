#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import xarray as xr
from esmlab.dataarray_statistics import (
    da_weighted_mean,
    da_weighted_sum,
    da_weighted_std,
)


@xr.register_dataarray_accessor("esm")
class EsmDataArrayAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._attrs = xarray_obj.attrs.copy()
        self._encoding = xarray_obj.encoding

    def update_attrs(self, da_output):
        for att in ["grid_loc", "coordinates"]:
            if att in self._attrs:
                del self._attrs[att]

        da_output.attrs = self._attrs
        da_output.encoding = {
            key: val
            for key, val in self._encoding.items()
            if key in ["_FillValue", "dtype"]
        }
        return da_output

    def weighted_sum(self, weights, dim=None):
        da_output = da_weighted_sum(self._obj, weights, dim=dim)
        return self.update_attrs(da_output)

    def weighted_mean(self, weights, dim=None, apply_nan_mask=True):
        da_output = da_weighted_mean(
            self._obj, weights, dim=dim, apply_nan_mask=apply_nan_mask
        )
        return self.update_attrs(da_output)

    def weighted_std(self, weights, dim=None, apply_nan_mask=True, ddof=0):
        da_output = da_weighted_std(
            self._obj, weights, dim=dim, apply_nan_mask=apply_nan_mask, ddof=ddof
        )
        return self.update_attrs(da_output)
