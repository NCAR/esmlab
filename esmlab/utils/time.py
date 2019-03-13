from __future__ import absolute_import, division, print_function

import uuid
from datetime import datetime

import cftime
import numpy as np
import xarray as xr

from .variables import get_variables

DEFAULT_TIME_UNITS = 'days since 0001-01-01 00:00:00'
DEFAULT_CALENDAR = 'standard'


class time_manager(object):
    """Class to support managing the time coordinate of datasets.
    """

    def __init__(self, ds, time_coord_name=None, year_offset=None):
        """
        Parameters
        ----------
        ds : xarray.Dataset
           The dataset on which to operate.
        time_coord_name : string, optional
           Name of the time coordinate of `ds`; if not provided, the
           code will attempt to infer it.
        """
        self._ds = ds.copy()
        self.orig_time_coord_name = None
        self.orig_time_coord_decoded = None
        self._time_computed = False

        if time_coord_name is None:
            self.time_coord_name = self._infer_time_coord_name()
        else:
            self.time_coord_name = time_coord_name
        self._infer_time_bound_var()

        self.year_offset = year_offset
        self.time_orig = self.time.copy()

    @property
    def time(self):
        return self._ds[self.time_coord_name]

    @property
    def time_attrs(self):
        """Get the attributes of the time coordinate.
        """
        attrs = self.time.attrs
        encoding = self.time.encoding

        if 'units' in attrs:
            units = attrs['units']
        elif 'units' in encoding:
            units = encoding['units']
        else:
            units = None

        if 'calendar' in attrs:
            calendar = attrs['calendar']
        elif 'calendar' in encoding:
            calendar = encoding['calendar']
        else:
            calendar = DEFAULT_CALENDAR

        if 'bounds' in attrs:
            bounds = attrs['bounds']
        elif 'bounds' in encoding:
            bounds = encoding['bounds']
        else:
            bounds = None

        return {'units': units, 'calendar': calendar, 'bounds': bounds}

    def _infer_time_coord_name(self):
        """Infer name for time coordinate in a dataset
        """

        if 'time' in self._ds.variables:
            return 'time'

        unlimited_dims = self._ds.encoding.get('unlimited_dims', None)
        if len(unlimited_dims) == 1:
            return list(unlimited_dims)[0]
        raise ValueError(
            'Cannot infer `time_coord_name` from multiple unlimited dimensions: %s \n\t\t ***** Please specify time_coord_name to use. *****'
            % unlimited_dims
        )

    def _infer_time_bound_var(self):
        """Infer time_bound variable in a dataset.
        """
        self.tb_name = self.time_attrs['bounds']
        self.tb_dim = None

        if self.tb_name:
            self.tb_dim = self._ds[self.tb_name].dims[1]

    def get_time_undecoded(self):
        """Return time undecoded.
        """
        if self.time.dtype != np.dtype('O'):
            if self.orig_time_coord_decoded is None:
                self.orig_time_coord_decoded = False
            return self.time

        if not self.time_attrs['units']:
            raise ValueError('cannot undecode time')

        # un-decode time
        time = xr.DataArray(self.time)
        time.values = cftime.date2num(
            self.time, units=self.time_attrs['units'], calendar=self.time_attrs['calendar']
        )
        if self.orig_time_coord_decoded is None:
            self.orig_time_coord_decoded = True
        return time

    def get_time_decoded(self, midpoint=True):
        """Return time decoded.
        """
        # to compute a time midpoint, we need a time_bound variable
        if midpoint and self.time_bound is None:
            raise ValueError('cannot compute time midpoint w/o time bounds')

        if midpoint:
            time_values = self.time_bound.mean(self.tb_dim)

        else:
            # if time has already been decoded and there's no year_offset,
            # just return the time as is
            if self.time.dtype == np.dtype('O'):
                if self.year_offset is None:
                    return time_values

                # if we need to un-decode time to apply the year_offset,
                # make sure there are units to do so with
                time_values = self.get_time_undecoded()

            # time has not been decoded
            else:
                time_values = self.time

        if self.year_offset is not None:
            time_values += cftime.date2num(
                datetime(int(self.year_offset), 1, 1),
                units=self.time_attrs['units'],
                calendar=self.time_attrs['calendar'],
            )

        time_out = xr.DataArray(self.time)
        time_out.values = xr.CFTimeIndex(
            cftime.num2date(
                time_values, units=self.time_attrs['units'], calendar=self.time_attrs['calendar']
            )
        )
        return time_out

    def compute_time(self, retain_orig_time_coord=True):
        """Compute the mid-point of the time bounds.
        """

        if self._time_computed:
            return self._ds

        if retain_orig_time_coord:
            self.orig_time_coord_name = 't' + uuid.uuid4().hex
            self._ds[self.orig_time_coord_name] = self.get_time_undecoded()

        if self.time_bound is not None:
            groupby_coord = self.get_time_decoded(midpoint=True)

        else:
            groupby_coord = self.get_time_decoded(midpoint=False)

        self._ds[self.time_coord_name].values = groupby_coord.values

        self._time_computed = True

        return self._ds

    def restore_dataset(self, ds=None):
        """Return the original time variable.
        """
        if not self._time_computed:
            raise ValueError('time was not computed; cannot restore dataset')
        time_values = ds[self.orig_time_coord_name].values
        if self.orig_time_coord_decoded:
            time_values = xr.CFTimeIndex(
                cftime.num2date(
                    time_values,
                    units=self.time_attrs['units'],
                    calendar=self.time_attrs['calendar'],
                )
            )
        ds[self.time_coord_name].values = time_values
        ds = ds.drop(self.orig_time_coord_name)
        return ds

    @property
    def time_bound(self):
        """return time bound, ensuring that it has not been decoded.
        """
        if self.tb_name is None:
            return

        if self._ds[self.tb_name].dtype == np.dtype('O'):
            tb_value = cftime.date2num(
                self._ds[self.tb_name],
                units=self.time_attrs['units'],
                calendar=self.time_attrs['calendar'],
            )
            return xr.DataArray(tb_value, dims=(self.time_coord_name, self.tb_dim))
        else:
            return self._ds[self.tb_name]

    @property
    def time_bound_diff(self):
        """Compute the difference between time bounds.
        """
        time_bound_diff = xr.ones_like(self.time, dtype=self.time.dtype)

        time_bound_diff.name = self.tb_name + '_diff'
        time_bound_diff.attrs = {}

        if self.time_bound is not None:
            time_bound_diff.values = self.time_bound.diff(dim=self.tb_dim)[:, 0]
            if self.tb_dim in time_bound_diff.coords:
                time_bound_diff = time_bound_diff.drop(self.tb_dim)

        return time_bound_diff


def time_year_to_midyeardate(ds, time_coord_name):
    """Set the time coordinate to the mid-point of the year.
    """
    ds[time_coord_name].values = np.array(
        [cftime.datetime(year, 7, 2) for year in ds[time_coord_name]]
    )
    return ds


def compute_time_var(ds, midpoint=True, year_offset=None):
    """Compute the time coordinate of a dataset.

    Parameters
    ----------
    ds : `xarray.Dataset`
       The dataset on which to operate
    midpoint : bool, optional [default=True]
       Return time at the midpoints of the `time:bounds`
    year_offset : numeric, optional
       Integer year by which to offset the time axis.

    Returns
    -------
    ds : `xarray.Dataset`
       The dataset with time coordinate modified.
    """
    tm = time_manager(ds, year_offset=year_offset)
    ds = tm.compute_time()
    ds[tm.time_coord_name] = tm.get_time_decoded(midpoint)
    ds = ds.drop(tm.orig_time_coord_name)
    return ds


def uncompute_time_var(ds):
    """Return time coordinate from object to float.

    Parameters
    ----------
    ds : `xarray.Dataset`
       The dataset on which to operate

    Returns
    -------
    ds : `xarray.Dataset`
       The dataset with time coordinate modified.
    """
    tm = time_manager(ds)
    ds = tm.compute_time()
    ds[tm.time_coord_name] = tm.get_time_undecoded()
    ds = ds.drop(tm.orig_time_coord_name)
    return ds


def sel_time(ds, indexer_val, year_offset=None):
    """Return dataset truncated to specified time range.

    Parameters
    ----------
    ds : `xarray.Dataset`
       The dataset on which to operate
    indexer_val : scalar, slice, or array_like
       value passed to ds.isel(**{time_coord_name: indexer_val})
    year_offset : numeric, optional
       Integer year by which to offset the time axis.

    Returns
    -------
    ds : `xarray.Dataset`
       The dataset with time coordinate truncated.
    """

    tm = time_manager(ds, year_offset=year_offset)
    ds = tm.compute_time()
    ds = ds.sel(**{tm.time_coord_name: indexer_val})
    return tm.restore_dataset(ds)
