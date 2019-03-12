#!/usr/bin/env python
"""Contains functions to compute climatologies."""
from __future__ import absolute_import, division, print_function

import numpy as np
import xarray as xr
import cftime as cft

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


@esmlab_xr_set_options(arithmetic_join="exact")
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
    time_dot_month = ".".join([time_coord_name, "month"])
    computed_dset = (
        dset.drop(static_variables)
        .groupby(time_dot_month)
        .mean(time_coord_name)
        .rename({"month": time_coord_name})
    )

    # Put static_variables back
    computed_dset = set_static_variables(computed_dset, dset, static_variables)

    # add month_bounds
    computed_dset["month"] = computed_dset[time_coord_name].copy()
    attrs["month"] = {"long_name": "Month", "units": "month"}
    encoding["month"] = {"dtype": "int32", "_FillValue": None}
    encoding[time_coord_name] = {"dtype": "float", "_FillValue": None}

    if "calendar" in attrs[time_coord_name]:
        attrs[time_coord_name]["calendar"] = attrs[time_coord_name]["calendar"]

    if tm.time_bound is not None:
        computed_dset[tm.tb_name] = tm.time_bound - tm.time_bound[0, 0]
        computed_dset[time_coord_name].values = computed_dset[tm.tb_name].mean(tm.tb_dim).values

        encoding[tm.tb_name] = {"dtype": "float", "_FillValue": None}
        attrs[tm.tb_name] = {"long_name": tm.tb_name, "units": "days since 0001-01-01 00:00:00"}

        attrs[time_coord_name] = {
            "long_name": time_coord_name,
            "units": "days since 0001-01-01 00:00:00",
            "bounds": tm.tb_name,
        }

        if "calendar" in attrs[time_coord_name]:
            attrs[tm.tb_name]["calendar"] = attrs[time_coord_name]["calendar"]

    # Put the attributes, encoding back
    computed_dset = set_metadata(computed_dset, attrs, encoding, additional_attrs={})

    computed_dset = tm.restore_dataset(computed_dset)

    return computed_dset


@esmlab_xr_set_options(arithmetic_join="exact")
def compute_mon_averages(dset, time_coord_name=None):
    """Calculates monthly averages from a more frequently recorded dataset

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

    static_variables = get_static_variables(dset, time_coord_name)

    # save metadata
    attrs, encoding = save_metadata(dset)


    if tm.time_bound is None:
        raise RuntimeError("Dataset must have time_bound variable to be able to"
                           "generate weighted monthly averages.")


    def date2mthIndex(date):
        """ return the month number of a given date """
        return date.year*12 + date.month
    def mthIndex2date(mth_index):
        """ return a datetime object for a given month index"""
        return cft.datetime((mth_index-1)//12, (mth_index-1)%12 +1, 1)

    # set of partially covered month indices that are to be dropped
    partial_mths = set()

    def weighted_monthly_mean(ds_grp,calendar):
        """ computes weighted averages of a given dataset group (ds_grp) """

        # determine the year and month of this group
        median_date = cft.num2date( ds_grp[tb_name].mean(),
                                    ds_grp[tb_name].attrs['units'],
                                    calendar)
        group_yr = median_date.year
        group_mth = median_date.month

        # begin date of this group (month):
        begin_date = cft.datetime(group_yr, group_mth, 1)
        begin_num = cft.date2num(begin_date,
                                 ds_grp[tb_name].attrs['units'],
                                 calendar)

        # end date of this group (month):
        end_date = cft.datetime(group_yr+(group_mth+1)//12, (group_mth+1)%13 + group_mth//12, 1)
        end_num = cft.date2num(end_date,
                               ds_grp[tb_name].attrs['units'],
                               calendar)

        # length of this group (e.g., number of days in this month):
        duration = end_num-begin_num
        nw = len(ds_grp[tb_name].data) # number of weights needed
        weights = [0.0]*nw

        if nw>1:
            # determine whether initial and/or last chunks are partly within this group
            initial_partly = ds_grp[tb_name].data[0] == ds_grp[tb_name].data[1]
            last_partly = ds_grp[tb_name].data[-2] == ds_grp[tb_name].data[-1]

            # compute the first weight:
            if not initial_partly:
                chunk_duration = ds_grp[tb_name].data[1] - ds_grp[tb_name].data[0]
                weights[0] =  0.5 * chunk_duration / duration
            else:
                chunk_duration = ds_grp[tb_name].data[0] - begin_num
                weights[0] =  chunk_duration / duration

            # compute the weights in the middle
            for i in range(1,nw-1):
                if ds_grp[tb_name].data[i] != ds_grp[tb_name].data[i-1]:
                    chunk_duration = ds_grp[tb_name].data[i] - ds_grp[tb_name].data[i-1]
                elif ds_grp[tb_name].data[i] != ds_grp[tb_name].data[i+1]:
                    chunk_duration = ds_grp[tb_name].data[i+1] - ds_grp[tb_name].data[i]
                weights[i] = 0.5 * chunk_duration / duration

            # compute the last weight:
            if not last_partly:
                chunk_duration = ds_grp[tb_name].data[-1] - ds_grp[tb_name].data[-2]
                weights[-1] =  0.5 * chunk_duration / duration
            else:
                chunk_duration = end_num - ds_grp[tb_name].data[-1]
                weights[-1] =  chunk_duration / duration

        else:
            pass # Partially covered month. Will be dropped.


        if sum(weights)<.9999:
            print("WARNING: partially covered month beginning on:", begin_date)
            print("\tDropping this month...")
            partial_mths.add(date2mthIndex(median_date))


        # convert weights to an xr.DataArray:
        weights = xr.DataArray( weights,
                                coords={"stacked_time_d2":ds_grp.stacked_time_d2},
                                dims=("stacked_time_d2",) )

        # instantiate the mean dataset. (TODO: make this more efficient)
        ds_mean = ds_grp.sum(dim="stacked_time_d2")

        # now compute the correct means for variables with time dimension
        for da in ds_mean.variables:
            if "stacked_time_d2" in ds_grp[da].dims:
                ds_mean[da] = xr.dot(ds_grp[da],weights)

        return ds_mean

    # Create a data array of time_bound months.
    # This data array is to be used when grouping dset.
    tb_name = tm.tb_name
    cal_name = dset[time_coord_name].attrs['calendar']
    tb_name_mth = tb_name+"_mth_index"
    tb_month = xr.DataArray(dset[tb_name], name=tb_name_mth)
    tb_month.data = [ [date2mthIndex(date0), date2mthIndex(date1)] \
                        for [date0,date1] in cft.num2date(dset[tb_name],
                                                          dset[tb_name].attrs['units'],
                                                          cal_name)]

    # Group by time_bound months and apply weighted averaging
    computed_dset = (
        dset.drop(static_variables)
        .groupby(tb_month)
        .apply(weighted_monthly_mean, calendar=cal_name)
        .rename({tb_name_mth: time_coord_name})
    )

    # drop partial months:
    computed_dset = computed_dset.drop(partial_mths, dim=time_coord_name)

    # correct time and time_bound
    ntime = len(computed_dset[time_coord_name])
    times = []
    time_bounds = []
    for t in range(ntime):
        begin_date  = mthIndex2date(computed_dset[time_coord_name][t])
        begin_num   = cft.date2num( begin_date,
                                    dset[time_coord_name].attrs['units'],
                                    cal_name)
        end_date    = mthIndex2date(computed_dset[time_coord_name][t]+1)
        end_num     = cft.date2num( end_date, 
                                    dset[time_coord_name].attrs['units'],
                                    cal_name)

        mean_num   = (end_num+begin_num)/2.0
        times.append(mean_num)
        time_bounds.append([begin_num,end_num])

    computed_dset[time_coord_name].data = times
    computed_dset.drop(tb_name)
    computed_dset[tb_name] = xr.DataArray(time_bounds,
                                          dims=(time_coord_name,'d2'),
                                          attrs={'units':dset[tb_name].attrs['units']})

    # Put static_variables back
    computed_dset = set_static_variables(computed_dset, dset, static_variables)

    # Put the attributes, encoding back
    computed_dset = set_metadata(computed_dset, attrs, encoding, additional_attrs={})

    return computed_dset


@esmlab_xr_set_options(arithmetic_join="exact")
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
    time_dot_month = ".".join([time_coord_name, "month"])
    if slice_mon_clim_time is None:
        computed_dset = dset.drop(static_variables).groupby(time_dot_month) - dset.drop(
            static_variables
        ).groupby(time_dot_month).mean(time_coord_name)
    else:
        computed_dset = dset.drop(static_variables).groupby(time_dot_month) - dset.drop(
            static_variables
        ).sel(time=slice_mon_clim_time).groupby(time_dot_month).mean(time_coord_name)

    # reset month to become a variable
    computed_dset = computed_dset.reset_coords("month")

    # Put static_variables back
    computed_dset = set_static_variables(computed_dset, dset, static_variables)

    # Put the attributes, encoding back
    computed_dset = set_metadata(
        computed_dset, attrs, encoding, additional_attrs={"month": {"long_name": "Month"}}
    )

    computed_dset = tm.restore_dataset(computed_dset)

    return computed_dset


@esmlab_xr_set_options(arithmetic_join="exact")
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

    time_dot_year = ".".join([time_coord_name, "year"])
    # Compute weights
    if weights:
        if len(weights) != len(dset[time_coord_name]):
            raise ValueError("weights and dataset time values must be of the same length")
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
        .rename({"year": time_coord_name})
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
        .rename({"year": time_coord_name})
    )
    ones_out = (
        (ones * weights)
        .groupby(time_dot_year)
        .sum(time_coord_name)
        .rename({"year": time_coord_name})
    )
    ones_out = ones_out.where(ones_out > 0.0)

    # Renormalize to appropriately account for missing values
    computed_dset = computed_dset / ones_out

    # Apply the valid-values mask
    for v in variables:
        computed_dset[v] = computed_dset[v].where(valid[v])

    # address time
    attrs[time_coord_name] = {
        "long_name": time_coord_name,
        "units": "days since 0001-01-01 00:00:00",
    }

    if "calendar" in attrs[time_coord_name]:
        attrs[time_coord_name]["calendar"] = attrs[time_coord_name]["calendar"]

    # compute the time_bound variable
    if tm.time_bound is not None:
        tb_out_lo = (
            tm.time_bound[:, 0]
            .groupby(time_dot_year)
            .min(dim=time_coord_name)
            .rename({"year": time_coord_name})
        )
        tb_out_hi = (
            tm.time_bound[:, 1]
            .groupby(time_dot_year)
            .max(dim=time_coord_name)
            .rename({"year": time_coord_name})
        )

        computed_dset[tm.tb_name] = xr.concat((tb_out_lo, tb_out_hi), dim=tm.tb_dim)
        attrs[time_coord_name]["bounds"] = tm.tb_name

    # Put static_variables back
    computed_dset = set_static_variables(computed_dset, dset, static_variables)

    # make year into date
    computed_dset = time_year_to_midyeardate(computed_dset, time_coord_name)

    # Put the attributes, encoding back
    computed_dset = set_metadata(computed_dset, attrs, encoding, additional_attrs={})

    computed_dset = tm.restore_dataset(computed_dset)

    return computed_dset
