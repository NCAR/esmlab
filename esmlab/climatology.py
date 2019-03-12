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
def compute_mon_climatology(dset, time_coord_name=None, weighted=False):
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


    if weighted and tm.time_bound is None:
        raise RuntimeError("Dataset must have time_bound variable to be able to"
                           "generate weighted monthly climatology.")


    def weighted_monthly_mean(ds,calendar):
        """ computes weighted averages of a given dataset group (ds) """

        # determine the year and month of this group
        median_date = cft.num2date( ds[tb_name].mean(), #ds[tb_name].median() #not yet implemented on dask
                                    ds[tb_name].attrs['units'],
                                    calendar)
        group_yr = median_date.year
        group_mth = median_date.month

        # begin date of this group (month):
        begin_date = cft.datetime(group_yr, group_mth, 1)
        begin_num = cft.date2num(begin_date,
                                 ds[tb_name].attrs['units'],
                                 calendar)

        # end date of this group (month):
        end_date = cft.datetime(group_yr+(group_mth+1)//12, (group_mth+1)%13 + group_mth//12, 1)
        end_num = cft.date2num(end_date,
                               ds[tb_name].attrs['units'],
                               calendar)

        # length of this group (e.g., number of days in this month):
        duration = end_num-begin_num
        nw = len(ds[tb_name].data) # number of weights needed
        weights = [0.0]*nw

        if nw>1:
            # determine whether initial and/or last chunks are partly within this group
            initial_partly = ds[tb_name].data[0] == ds[tb_name].data[1]
            last_partly = ds[tb_name].data[-2] == ds[tb_name].data[-1]

            # compute the first weight:
            if not initial_partly:
                chunk_duration = ds[tb_name].data[1] - ds[tb_name].data[0]
                weights[0] =  0.5 * chunk_duration / duration
            else:
                chunk_duration = ds[tb_name].data[0] - begin_num
                weights[0] =  chunk_duration / duration

            # compute the weights in the middle
            for i in range(1,nw-1):
                if ds[tb_name].data[i] != ds[tb_name].data[i-1]:
                    chunk_duration = ds[tb_name].data[i] - ds[tb_name].data[i-1]
                elif ds[tb_name].data[i] != ds[tb_name].data[i+1]:
                    chunk_duration = ds[tb_name].data[i+1] - ds[tb_name].data[i]
                weights[i] = 0.5 * chunk_duration / duration

            # compute the last weight:
            if not last_partly:
                chunk_duration = ds[tb_name].data[-1] - ds[tb_name].data[-2]
                weights[-1] =  0.5 * chunk_duration / duration
            else:
                chunk_duration = end_num - ds[tb_name].data[-1]
                weights[-1] =  chunk_duration / duration

        else:
            pass # TODO



        if sum(weights)<.9999:
            print("WARNING: missing slice for the month beginning on:", begin_date)
            # TODO: proper way of handling this ?


        # convert weights to an xr.DataArray:
        weights = xr.DataArray( weights,
                                coords={"stacked_time_d2":ds.stacked_time_d2},
                                dims=("stacked_time_d2",) )

        # instantiate the mean dataset. (TODO: make this more efficient)
        ds_mean = ds.sum(dim="stacked_time_d2")

        # now compute the correct means for variables with time dimension
        for da in ds_mean.variables:
            if "stacked_time_d2" in ds[da].dims:
                ds_mean[da] = xr.dot(ds[da],weights)

        return ds_mean

    # Compute climatology
    if weighted:
        # Create a data array of time_bound months.
        # This data array is to be used when grouping dset.
        tb_name = tm.tb_name
        tb_name_mth = tb_name+"_month"
        tb_month = xr.DataArray(dset[tb_name], name=tb_name_mth)
        tb_month.data = [ [date0.month,date1.month] for [date0,date1] in
                            cft.num2date(dset[tb_name],
                                         dset[tb_name].attrs['units'],
                                         dset[time_coord_name].attrs['calendar'])]

        # Group by time_bound months and apply weighted averaging
        computed_dset = (
            dset.drop(static_variables)
            .groupby(tb_month)
            .apply(weighted_monthly_mean, calendar=dset[time_coord_name].attrs['calendar'])
            .rename({tb_name_mth: time_coord_name})
        )

    else:
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

    if weighted:
        pass # TODO
    else:
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

    if weighted:
        pass # TODO
    else:
        computed_dset = tm.restore_dataset(computed_dset)

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
