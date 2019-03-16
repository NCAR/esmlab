from __future__ import absolute_import, division, print_function

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
        self._ds = ds
        self.year_offset = year_offset
        self.time_bound_diff = None
        self.time_orig_decoded = None
        self._set_time(time_coord_name)

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

        key_attrs = {'units': units, 'calendar': calendar, 'bounds': bounds}
        z = attrs.copy()
        z.update(key_attrs)
        return z

    @property
    def time_bound_attrs(self):
        """Get the attributes of the time coordinate.
        """

        if self.time_bound is None:
            return {}
        attrs = self._ds[self.tb_name].attrs
        key_attrs = self.time_attrs
        z = attrs.copy()
        z.update(key_attrs)
        return z

    def _set_time(self, time_coord_name):
        """store the original time and time_bound data from the dataset;
           ensure that time_bound, if present, is not decoded.
        """

        self._infer_time_coord_name(time_coord_name)
        self.time = self._ds[self.time_coord_name].copy()
        self.time_orig_decoded = self.isdecoded(self.time)

        self._infer_time_bound_var()
        if self.tb_name is None:
            self.time_bound = None

        else:
            self.time_bound = self._ds[self.tb_name].copy()
            if self.isdecoded(self._ds[self.tb_name]):
                tb_data = cftime.date2num(
                    self._ds[self.tb_name],
                    units=self.time_attrs['units'],
                    calendar=self.time_attrs['calendar'],
                )
                self.time_bound.data = tb_data

    def _compute_time_bound_diff(self, ds):
        """Compute the difference between time bounds.
        """
        time_bound_diff = xr.ones_like(ds[self.time_coord_name], dtype=np.float64)

        if self.time_bound is not None:
            time_bound_diff.name = self.tb_name + '_diff'
            time_bound_diff.attrs = {}
            time_bound_diff.data = self.time_bound.diff(dim=self.tb_dim)[:, 0]
            if self.tb_dim in time_bound_diff.coords:
                time_bound_diff = time_bound_diff.drop(self.tb_dim)

        return time_bound_diff

    def _infer_time_coord_name(self, time_coord_name):
        """Infer name for time coordinate in a dataset
        """
        if time_coord_name:
            self.time_coord_name = time_coord_name

        elif 'time' in self._ds.variables:
            self.time_coord_name = 'time'

        else:
            unlimited_dims = self._ds.encoding.get('unlimited_dims', None)
            if len(unlimited_dims) == 1:
                self.time_coord_name = list(unlimited_dims)[0]
            else:
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

    def isdecoded(self, obj):
        return obj.dtype == np.dtype('O')

    def get_time_undecoded(self):
        """Return time undecoded.
        """
        time = self.time.copy()

        if not self.isdecoded(time):
            return time

        if not self.time_attrs['units']:
            raise ValueError('cannot undecode time')

        # un-decode time
        time.data = cftime.date2num(
            time, units=self.time_attrs['units'], calendar=self.time_attrs['calendar']
        )

        return time

    def get_time_decoded(self, midpoint=True):
        """Return time decoded.
        """
        # to compute a time midpoint, we need a time_bound variable
        if midpoint and self.time_bound is None:
            raise ValueError('cannot compute time midpoint w/o time bounds')

        if midpoint:
            time_data = self.time_bound.mean(self.tb_dim)

        else:
            # if time has already been decoded and there's no year_offset,
            # just return the time as is
            if self.isdecoded(self.time):
                if self.year_offset is None:
                    return self.time

                # if we need to un-decode time to apply the year_offset,
                time_data = self.get_time_undecoded()

            # time has not been decoded
            else:
                time_data = self.time

        if self.year_offset is not None:
            time_data += cftime.date2num(
                datetime(int(self.year_offset), 1, 1),
                units=self.time_attrs['units'],
                calendar=self.time_attrs['calendar'],
            )

        time_out = self.time.copy()
        time_out.data = xr.CFTimeIndex(
            cftime.num2date(
                time_data, units=self.time_attrs['units'], calendar=self.time_attrs['calendar']
            )
        )
        return time_out

    def compute_time(self):
        """Compute the mid-point of the time bounds.
        """

        ds = self._ds.copy()

        if self.time_bound is not None:
            groupby_coord = self.get_time_decoded(midpoint=True)

        else:
            groupby_coord = self.get_time_decoded(midpoint=False)

        ds[self.time_coord_name].data = groupby_coord.data

        if self.time_bound is not None:
            ds[self.tb_name].data = self.time_bound.data
            self.time_bound[self.time_coord_name].data = groupby_coord.data
        self.time_bound_diff = self._compute_time_bound_diff(ds)

        return ds

    def restore_dataset(self, ds):
        """Return the original time variable to decoded or undecoded state.
        """

        # get the time data from dataset
        time_data = ds[self.time_coord_name].data

        # if time was not originally decoded, return the dataset with time
        # un-decoded
        if not self.time_orig_decoded and self.isdecoded(time_data):
            time_data = cftime.date2num(
                time_data, units=self.time_attrs['units'], calendar=self.time_attrs['calendar']
            )

        ds[self.time_coord_name].attrs = self.time_attrs
        ds[self.time_coord_name].data = time_data.astype(self.time.dtype)
        if self.time_bound is not None:
            ds[self.tb_name].attrs = self.time_bound_attrs

        return ds


def time_year_to_midyeardate(ds, time_coord_name):
    """Set the time coordinate to the mid-point of the year.
    """
    ds[time_coord_name].data = np.array(
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
