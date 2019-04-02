from __future__ import absolute_import, division, print_function

from datetime import datetime

import cftime
import dask.array as da
import numpy as np
import xarray as xr

from .common_utils import esmlab_xr_set_options


@xr.register_dataset_accessor('esmlab')
class EsmlabAccessor(object):
    def __init__(self, dset):
        self._ds = dset
        self._ds_time_computed = None
        self.time = None
        self.year_offset = None
        self.time_bound_diff = None
        self.time_orig_decoded = None
        self.time_coord_name = None
        self.tb_name = None
        self.tb_dim = None
        self.time_bound = None
        self.static_variables = []
        self.variables = []
        self._set_time()
        self._get_variables()
        self.compute_time()
        self._get_original_metadata()

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
            calendar = 'standard'

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

    def _set_time(self):
        """store the original time and time_bound data from the dataset;
           ensure that time_bound, if present, is not decoded.
        """
        self._infer_time_coord_name()
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
        time = self.time.copy()
        if not self.isdecoded(time):
            return time

        if not self.time_attrs['units']:
            raise ValueError('Cannot undecode time')

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

    def _infer_time_coord_name(self):
        """Infer name for time coordinate in a dataset
        """
        if 'time' in self._ds.variables:
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

    def compute_time(self):
        """Compute the mid-point of the time bounds.
        """

        ds = self._ds.copy(deep=True)

        if self.time_bound is not None:
            groupby_coord = self.get_time_decoded(midpoint=True)

        else:
            groupby_coord = self.get_time_decoded(midpoint=False)

        ds[self.time_coord_name].data = groupby_coord.data

        if self.time_bound is not None:
            ds[self.tb_name].data = self.time_bound.data
            self.time_bound[self.time_coord_name].data = groupby_coord.data
        self.time_bound_diff = self._compute_time_bound_diff(ds)

        self._ds_time_computed = ds

    def _get_variables(self):
        if not self.static_variables:
            self.static_variables = [
                v for v in self._ds.variables if self.time_coord_name not in self._ds[v].dims
            ]

        if not self.variables:
            self.variables = [
                v
                for v in self._ds.variables
                if self.time_coord_name in self._ds[v].dims
                and v not in [self.time_coord_name, self.tb_name]
            ]

    def time_year_to_midyeardate(self):
        """Set the time coordinate to the mid-point of the year.
        """
        ds = self._ds_time_computed.copy(True)
        ds[self.time_coord_name].data = np.array(
            [cftime.datetime(entry.year, 7, 2) for entry in ds[self.time_coord_name].data]
        )
        return ds

    def compute_time_var(self, midpoint=True, year_offset=None):
        """Compute the time coordinate of a dataset.

        Parameters
        ----------
        midpoint : bool, optional [default=True]
                Return time at the midpoints of the `time:bounds`
        year_offset : numeric, optional
        Integer year by which to offset the time axis.

        Returns
        -------
        ds : `xarray.Dataset`
        The dataset with time coordinate modified.
        """
        self.year_offset = year_offset
        ds = self._ds_time_computed.copy(True)
        ds[self.time_coord_name] = self.get_time_decoded(midpoint)
        return ds

    def uncompute_time_var(self):
        """Return time coordinate from object to float.
        Returns
        -------
        ds : `xarray.Dataset`
        The dataset with time coordinate modified.
        """
        ds = self._ds_time_computed.copy(True)
        ds[self.time_coord_name] = self.get_time_undecoded()
        return ds

    def sel_time(self, indexer_val, year_offset=None):
        """Return dataset truncated to specified time range.

        Parameters
        ----------

        indexer_val : scalar, slice, or array_like
            value passed to ds.isel(**{time_coord_name: indexer_val})
        year_offset : numeric, optional
            Integer year by which to offset the time axis.

        Returns
        -------
        ds : `xarray.Dataset`
        The dataset with time coordinate truncated.
        """
        self.year_offset = year_offset
        self.compute_time()
        ds = self._ds_time_computed.copy(True)
        ds = ds.sel(**{self.time_coord_name: indexer_val})
        return ds

    def _get_original_metadata(self):
        self._attrs = {v: self._ds[v].attrs for v in self.variables}
        self._encoding = {
            v: {
                key: val
                for key, val in self._ds[v].encoding.items()
                if key in ['dtype', '_FillValue', 'missing_value']
            }
            for v in self.variables
        }

    def update_metadata(self, ds, new_attrs={}, new_encoding={}):

        attrs = self._attrs.copy()
        attrs.update(new_attrs)
        encoding = self._encoding.copy()
        encoding.update(new_encoding)

        for v in self.variables:
            ds[v].attrs = attrs[v]

            if v in encoding:
                if ds[v].dtype == 'int64':  # int64 breaks some other tools
                    encoding[v]['dtype'] = 'int32'

                ds[v].encoding = encoding[v]

        return ds

    def restore_dataset(self, ds, attrs={}, encoding={}):
        """Return the original time variable to decoded or undecoded state.
        """
        ds = xr.merge(
            (
                ds,
                self._ds_time_computed.drop(
                    [v for v in self._ds.variables if v not in self.static_variables]
                ),
            )
        )
        # get the time data from dataset
        time_data = ds[self.time_coord_name].data

        # if time was not originally decoded, return the dataset with time
        # un-decoded
        if not self.time_orig_decoded and self.isdecoded(time_data):
            time_data = cftime.date2num(
                time_data, units=self.time_attrs['units'], calendar=self.time_attrs['calendar']
            )

        elif self.time_orig_decoded and not self.isdecoded(time_data):
            time_data = cftime.num2date(
                time_data, units=self.time_attrs['units'], calendar=self.time_attrs['calendar']
            )

        ds[self.time_coord_name].attrs = self.time_attrs
        ds[self.time_coord_name].data = time_data.astype(self.time.dtype)
        if self.time_bound is not None:
            ds[self.tb_name].attrs = self.time_bound_attrs

        return self.update_metadata(ds, new_attrs=attrs, new_encoding=encoding)

    @esmlab_xr_set_options(arithmetic_join='exact')
    def mon_climatology(self):
        """ Calculates monthly climatology """

        time_dot_month = '.'.join([self.time_coord_name, 'month'])
        computed_dset = (
            self._ds_time_computed.drop(self.static_variables)
            .groupby(time_dot_month)
            .mean(self.time_coord_name)
            .rename({'month': self.time_coord_name})
        )
        computed_dset['month'] = computed_dset[self.time_coord_name].copy()
        attrs = {'month': {'long_name': 'Month', 'units': 'month'}}
        encoding = {
            'month': {'dtype': 'int32', '_FillValue': None},
            self.time_coord_name: {'dtype': 'float', '_FillValue': None},
        }

        if self.time_bound is not None:
            time_data = computed_dset[self.tb_name] - computed_dset[self.tb_name][0, 0]
            computed_dset[self.tb_name] = time_data
            computed_dset[self.time_coord_name].data = (
                computed_dset[self.tb_name].mean(self.tb_dim).data
            )
            encoding[self.tb_name] = {'dtype': 'float', '_FillValue': None}
        return self.restore_dataset(computed_dset, attrs=attrs, encoding=encoding)

    @esmlab_xr_set_options(arithmetic_join='exact')
    def mon_anomaly(self, slice_mon_clim_time=None):
        """ Calculates monthly anomaly

        Parameters
        ----------
        slice_mon_clim_time : slice, optional
                          a slice object passed to
                          `dset.isel(time=slice_mon_clim_time)` for subseting
                          the time-period overwhich the climatology is computed
        """
        time_dot_month = '.'.join([self.time_coord_name, 'month'])
        actual = self._ds_time_computed.drop(self.static_variables).groupby(time_dot_month)
        if slice_mon_clim_time:
            climatology = (
                self._ds_time_computed.drop(self.static_variables)
                .sel({self.time_coord_name: slice_mon_clim_time})
                .groupby(time_dot_month)
                .mean(self.time_coord_name)
            )
        else:
            climatology = (
                self._ds_time_computed.drop(self.static_variables)
                .groupby(time_dot_month)
                .mean(self.time_coord_name)
            )

        computed_dset = actual - climatology

        # reset month to become a variable
        computed_dset = computed_dset.reset_coords('month')

        computed_dset[self.time_coord_name].data = self.time.data
        if self.time_bound is not None:
            computed_dset[self.tb_name].data = self.time_bound.data

        attrs = {'month': {'long_name': 'Month'}}
        return self.restore_dataset(computed_dset, attrs=attrs)

    @esmlab_xr_set_options(arithmetic_join='exact')
    def ann_mean(self, weights=None):
        """ Calculates annual mean """
        time_dot_year = '.'.join([self.time_coord_name, 'year'])

        if isinstance(weights, xr.DataArray):
            data = weights.data

        elif isinstance(weights, (list, np.ndarray, da.Array)):
            data = weights

        else:
            data = self.time_bound_diff

        wgts = xr.ones_like(self.time_bound_diff)
        wgts.data = data
        wgts = wgts.groupby(time_dot_year) / wgts.groupby(time_dot_year).sum(xr.ALL_DIMS)
        wgts = wgts.rename('weights')
        groups = len(wgts.groupby(time_dot_year).groups)
        np.testing.assert_allclose(wgts.groupby(time_dot_year).sum(xr.ALL_DIMS), np.ones(groups))

        dset = self._ds_time_computed.drop(self.static_variables) * wgts

        def weighted_mean_arr(darr, wgts=None):
            # if NaN are present, we need to use individual weights
            total_weights = wgts.where(darr.notnull()).sum(dim=self.time_coord_name)
            return (
                darr.resample({self.time_coord_name: 'A'}).mean(dim=self.time_coord_name)
                / total_weights
            )

        ds_resample_mean = dset.apply(weighted_mean_arr, wgts=wgts)

        # ds_resample_mean = xr.Dataset()
        # for dvar in self.variables:
        #     darr = dset[dvar]
        #     # if NaN are present, we need to use individual weights
        #     if ~darr.notnull().all():
        #         total_weights = wgts.where(darr.notnull()).sum(dim=self.time_coord_name)
        #     else:
        #         total_weights = wgts.sum(dim=self.time_coord_name)

        #     ds_resample_mean[dvar] = darr.resample({self.time_coord_name:'A'}).mean(dim=self.time_coord_name) / total_weights

        mid_time = wgts[self.time_coord_name].groupby(time_dot_year).mean()
        ds_resample_mean[self.time_coord_name].data = mid_time.data
        # return self.restore_dataset(ds_resample_mean)
        return ds_resample_mean


def mon_climatology(dset):
    """Calculates monthly climatology

    Parameters
    ----------
    dset : xarray.Dataset
           The data on which to operate

    Returns
    -------
    computed_dset : xarray.Dataset
                    The computed monthly climatology data

    """

    return dset.esmlab.mon_climatology()


def mon_anomaly(dset, slice_mon_clim_time=None):
    """Calculates monthly anomaly

    Parameters
    ----------
    dset : xarray.Dataset
           The data on which to operate

    slice_mon_clim_time : slice, optional
                          a slice object passed to
                          `dset.isel(time=slice_mon_clim_time)` for subseting
                          the time-period overwhich the climatology is computed

    Returns
    -------
    computed_dset : xarray.Dataset
                    The computed monthly anomaly data

    """

    return dset.esmlab.mon_anomaly(slice_mon_clim_time=slice_mon_clim_time)


def ann_mean(dset, weights=None):
    """Calculates annual means

    Parameters
    ----------
    dset : xarray.Dataset
           The data on which to operate

    weights : array_like, optional
              weights to use for each time period.
              If None and dataset doesn't have `time_bound` variable,
              every time period has equal weight of 1.

    Returns
    -------
    computed_dset : xarray.Dataset
                    The computed annual mean data

    """

    return dset.esmlab.ann_mean(weights=weights)
