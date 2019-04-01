#!/usr/bin/env python
"""Contains functions to compute climatologies."""
from __future__ import absolute_import, division, print_function

import cftime as cft
import numpy as np
import xarray as xr

from .utils.common import esmlab_xr_set_options
from .utils.time import time_manager, time_year_to_midyeardate
from .utils.variables import (
    get_original_attrs,
    get_static_variables,
    get_variables,
    save_metadata,
    set_metadata,
    set_static_variables,
    update_attrs,
)


@esmlab_xr_set_options(arithmetic_join='exact')
def compute_mon_climatology(dset, time_coord_name=None):
    """Calculates monthly climatology (monthly means)

    Parameters
    ----------
    dset : xarray.Dataset
           The data on which to operate

    time_coord_name : string
            Name for time coordinate

    Returns
    -------
    computed_dset : xarray.Dataset
                    The computed monthly climatology data

    """

    tm = time_manager(dset, time_coord_name)
    dset = tm.compute_time()
    time_coord_name = tm.time_coord_name

    static_variables = get_static_variables(dset, time_coord_name)

    # save metadata
    attrs, encoding = save_metadata(dset)

    # Compute climatology
    time_dot_month = '.'.join([time_coord_name, 'month'])
    computed_dset = (
        dset.drop(static_variables)
        .groupby(time_dot_month)
        .mean(time_coord_name)
        .rename({'month': time_coord_name})
    )

    # Put static_variables back
    computed_dset = set_static_variables(computed_dset, dset, static_variables)

    # add month_bounds
    computed_dset['month'] = computed_dset[time_coord_name].copy()
    attrs['month'] = {'long_name': 'Month', 'units': 'month'}
    encoding['month'] = {'dtype': 'int32', '_FillValue': None}
    encoding[time_coord_name] = {'dtype': 'float', '_FillValue': None}

    if 'calendar' in attrs[time_coord_name]:
        attrs[time_coord_name]['calendar'] = attrs[time_coord_name]['calendar']

    if tm.time_bound is not None:
        computed_dset[tm.tb_name] = tm.time_bound - tm.time_bound[0, 0]
        computed_dset[time_coord_name].values = computed_dset[tm.tb_name].mean(tm.tb_dim).values

        encoding[tm.tb_name] = {'dtype': 'float', '_FillValue': None}
        attrs[tm.tb_name] = {'long_name': tm.tb_name, 'units': 'days since 0001-01-01 00:00:00'}

        attrs[time_coord_name] = {
            'long_name': time_coord_name,
            'units': 'days since 0001-01-01 00:00:00',
            'bounds': tm.tb_name,
        }

        if 'calendar' in attrs[time_coord_name]:
            attrs[tm.tb_name]['calendar'] = attrs[time_coord_name]['calendar']

    # Put the attributes, encoding back
    computed_dset = set_metadata(computed_dset, attrs, encoding, additional_attrs={})

    computed_dset = tm.restore_dataset(computed_dset)

    return computed_dset


@esmlab_xr_set_options(arithmetic_join='exact')
def compute_mon_means(dset, time_coord_name=None):
    """Calculates monthly averages of a dataset

    Parameters
    ----------
    dset : xarray.Dataset
           The data on which to operate

    time_coord_name : string
            Name for time coordinate

    Returns
    -------
    computed_dset : xarray.Dataset
                    The computed monthly average data

    """

    tm = time_manager(dset, time_coord_name)
    dset = tm.compute_time()
    time_coord_name = tm.time_coord_name
    tb_name = tm.tb_name
    cal_name = dset[time_coord_name].attrs['calendar']

    static_variables = get_static_variables(dset, time_coord_name)

    # save metadata
    attrs, encoding = save_metadata(dset)

    if tm.time_bound is None:
        raise RuntimeError(
            'Dataset must have time_bound variable to be able to'
            'generate weighted monthly averages.'
        )

    # extrapolate dset to begin time (time_bound[0][0]):
    # (without extrapolation, resampling is applied only in between the midpoints of first and last
    # time bounds since time is midpoint of time_bounds)
    tbegin_decoded = time_manager.decode_time(
        dset[tb_name].isel({time_coord_name: 0, tm.tb_dim: 0}),
        units=tm.time_attrs['units'],
        calendar=cal_name,
    )
    dset_begin = dset.isel({time_coord_name: 0})
    dset_begin[time_coord_name] = tbegin_decoded

    # extrapolate dset to end time (time_bound[-1][1]):
    # (because time is midpoint)
    tend_decoded = time_manager.decode_time(
        dset[tb_name].isel({time_coord_name: -1, tm.tb_dim: 1}),
        units=tm.time_attrs['units'],
        calendar=cal_name,
    )
    dset_end = dset.isel({time_coord_name: -1})
    dset_end[time_coord_name] = tend_decoded

    # concatenate dset:
    computed_dset = xr.concat([dset_begin, dset, dset_end], dim=time_coord_name)

    # compute monthly means
    time_dot_month = '.'.join([time_coord_name, 'month'])
    computed_dset = (
        computed_dset.resample({time_coord_name: '1D'})  # resample to daily
        .nearest()  # get nearest (since time is midpoint)
        .isel({time_coord_name: slice(0, -1)})  # drop the last day: the end time
        .groupby(time_dot_month)  # group by month
        .mean(time_coord_name)  # monthly means
        .rename({'month': time_coord_name})
    )

    # Put static_variables back
    computed_dset = set_static_variables(computed_dset, dset, static_variables)

    # add month_bounds
    computed_dset['month'] = computed_dset[time_coord_name].copy()
    attrs['month'] = {'long_name': 'Month', 'units': 'month'}
    encoding['month'] = {'dtype': 'int32', '_FillValue': None}
    encoding[time_coord_name] = {'dtype': 'float', '_FillValue': None}

    def month2date(mth_index, begin_datetime):
        """ return a datetime object for a given month index"""
        mth_index += begin_datetime.year * 12 + begin_datetime.month
        return cft.datetime((mth_index - 1) // 12, (mth_index - 1) % 12 + 1, 1)

    # Correct time bounds:
    for m in range(len(computed_dset['month'])):
        computed_dset[tm.tb_name].values[m] = [
            # month begin date:
            cft.date2num(
                month2date(m, tbegin_decoded),
                units=attrs[time_coord_name]['units'],
                calendar=cal_name,
            ),
            # month end date:
            cft.date2num(
                month2date(m + 1, tbegin_decoded),
                units=attrs[time_coord_name]['units'],
                calendar=cal_name,
            ),
        ]

    encoding[tm.tb_name] = {'dtype': 'float', '_FillValue': None}
    attrs[tm.tb_name] = {'long_name': tm.tb_name, 'units': 'days since 0001-01-01 00:00:00'}

    attrs[time_coord_name] = {
        'long_name': time_coord_name,
        'units': 'days since 0001-01-01 00:00:00',
        'bounds': tm.tb_name,
    }

    attrs[time_coord_name]['calendar'] = cal_name
    attrs[tm.tb_name]['calendar'] = cal_name

    # Put the attributes, encoding back
    computed_dset = set_metadata(computed_dset, attrs, encoding, additional_attrs={})

    computed_dset = tm.restore_dataset(computed_dset)

    return computed_dset


@esmlab_xr_set_options(arithmetic_join='exact')
def compute_mon_anomaly(dset, slice_mon_clim_time=None, time_coord_name=None):
    """Calculates monthly anomaly

    Parameters
    ----------
    dset : xarray.Dataset
           The data on which to operate

    slice_mon_clim_time : slice, optional
                          a slice object passed to
                          `dset.isel(time=slice_mon_clim_time)` for subseting
                          the time-period overwhich the climatology is computed
    time_coord_name : string
            Name for time coordinate

    Returns
    -------
    computed_dset : xarray.Dataset
                    The computed monthly anomaly data

    """

    tm = time_manager(dset, time_coord_name)
    dset = tm.compute_time()
    time_coord_name = tm.time_coord_name

    static_variables = get_static_variables(dset, time_coord_name)

    # save metadata
    attrs, encoding = save_metadata(dset)

    # Compute anomaly
    time_dot_month = '.'.join([time_coord_name, 'month'])
    if slice_mon_clim_time is None:
        computed_dset = dset.drop(static_variables).groupby(time_dot_month) - dset.drop(
            static_variables
        ).groupby(time_dot_month).mean(time_coord_name)
    else:
        computed_dset = dset.drop(static_variables).groupby(time_dot_month) - dset.drop(
            static_variables
        ).sel(time=slice_mon_clim_time).groupby(time_dot_month).mean(time_coord_name)

    # reset month to become a variable
    computed_dset = computed_dset.reset_coords('month')

    # Put static_variables back
    computed_dset = set_static_variables(computed_dset, dset, static_variables)

    # Put the attributes, encoding back
    computed_dset = set_metadata(
        computed_dset, attrs, encoding, additional_attrs={'month': {'long_name': 'Month'}}
    )

    computed_dset = tm.restore_dataset(computed_dset)

    return computed_dset


@esmlab_xr_set_options(arithmetic_join='exact')
def compute_ann_mean(dset, weights=None, time_coord_name=None):
    """Calculates annual climatology (annual means)

    Parameters
    ----------
    dset : xarray.Dataset
           The data on which to operate

    weights : array_like, optional
              weights to use for each time period.
              If None and dataset doesn't have `time_bound` variable,
              every time period has equal weight of 1.

    time_coord_name : string
            Name for time coordinate

    Returns
    -------
    computed_dset : xarray.Dataset
                    The computed annual climatology data

    """

    tm = time_manager(dset, time_coord_name)
    dset = tm.compute_time()
    time_coord_name = tm.time_coord_name

    static_variables = get_static_variables(dset, time_coord_name)
    variables = get_variables(dset, time_coord_name, tm.tb_name)
    # save metadata
    attrs, encoding = save_metadata(dset)

    time_dot_year = '.'.join([time_coord_name, 'year'])
    # Compute weights
    if weights:
        if len(weights) != len(dset[time_coord_name]):
            raise ValueError('weights and dataset time values must be of the same length')
        else:
            dt = xr.ones_like(dset[time_coord_name], dtype=bool)
            dt.values = weights
            weights = dt / dt.sum(xr.ALL_DIMS)
            np.testing.assert_allclose(weights.sum(xr.ALL_DIMS), 1.0)

    elif not weights:
        dt = tm.time_bound_diff

        weights = dt.groupby(time_dot_year) / dt.groupby(time_dot_year).sum(xr.ALL_DIMS)

        np.testing.assert_allclose(weights.groupby(time_dot_year).sum(xr.ALL_DIMS), 1.0)

    # groupby.sum() does not seem to handle missing values correctly: yields 0 not nan
    # the groupby.mean() does return nans, so create a mask of valid values
    # for each variable
    valid = {
        v: dset[v]
        .groupby(time_dot_year)
        .mean(dim=time_coord_name)
        .notnull()
        .rename({'year': time_coord_name})
        for v in variables
    }
    ones = (
        dset.drop(static_variables)
        .where(dset.drop(static_variables).isnull())
        .fillna(1.0)
        .where(dset.drop(static_variables).notnull())
        .fillna(0.0)
    )

    # Compute annual mean
    computed_dset = (
        (dset.drop(static_variables) * weights)
        .groupby(time_dot_year)
        .sum(time_coord_name)
        .rename({'year': time_coord_name})
    )
    ones_out = (
        (ones * weights)
        .groupby(time_dot_year)
        .sum(time_coord_name)
        .rename({'year': time_coord_name})
    )
    ones_out = ones_out.where(ones_out > 0.0)

    # Renormalize to appropriately account for missing values
    computed_dset = computed_dset / ones_out

    # Apply the valid-values mask
    for v in variables:
        computed_dset[v] = computed_dset[v].where(valid[v])

    # address time
    attrs[time_coord_name] = {
        'long_name': time_coord_name,
        'units': 'days since 0001-01-01 00:00:00',
    }

    if 'calendar' in attrs[time_coord_name]:
        attrs[time_coord_name]['calendar'] = attrs[time_coord_name]['calendar']

    # compute the time_bound variable
    if tm.time_bound is not None:
        tb_out_lo = (
            tm.time_bound[:, 0]
            .groupby(time_dot_year)
            .min(dim=time_coord_name)
            .rename({'year': time_coord_name})
        )
        tb_out_hi = (
            tm.time_bound[:, 1]
            .groupby(time_dot_year)
            .max(dim=time_coord_name)
            .rename({'year': time_coord_name})
        )

        computed_dset[tm.tb_name] = xr.concat((tb_out_lo, tb_out_hi), dim=tm.tb_dim)
        attrs[time_coord_name]['bounds'] = tm.tb_name

    # Put static_variables back
    computed_dset = set_static_variables(computed_dset, dset, static_variables)

    # make year into date
    computed_dset = time_year_to_midyeardate(computed_dset, time_coord_name)

    # Put the attributes, encoding back
    computed_dset = set_metadata(computed_dset, attrs, encoding, additional_attrs={})

    computed_dset = tm.restore_dataset(computed_dset)

    return computed_dset
