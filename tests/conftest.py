import sys

import cftime
import numpy as np
import numpy.matlib as npm
import pytest
import xarray as xr


@pytest.fixture
def dset():
    start_date = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.float64)
    start_date = np.append(start_date, start_date + 365)
    end_date = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365], dtype=np.float64)
    end_date = np.append(end_date, end_date + 365)
    ds = xr.Dataset(coords={'time': 24, 'lat': 2, 'lon': 2, 'd2': 2})
    ds['time'] = xr.DataArray(end_date, dims='time')
    ds['lat'] = xr.DataArray([0, 1], dims='lat')
    ds['lon'] = xr.DataArray([0, 1], dims='lon')
    ds['d2'] = xr.DataArray([0, 1], dims='d2')
    ds['time_bound'] = xr.DataArray(
        np.array([start_date, end_date]).transpose(), dims=['time', 'd2']
    )
    ds['variable_1'] = xr.DataArray(
        np.append(
            np.zeros([12, 2, 2], dtype='float32'), np.ones([12, 2, 2], dtype='float32'), axis=0
        ),
        dims=['time', 'lat', 'lon'],
    )
    ds['variable_2'] = xr.DataArray(
        np.append(
            np.ones([12, 2, 2], dtype='float32'), np.zeros([12, 2, 2], dtype='float32'), axis=0
        ),
        dims=['time', 'lat', 'lon'],
    )
    ds.time.attrs['units'] = 'days since 0001-01-01 00:00:00'
    ds.time.attrs['calendar'] = 'noleap'
    ds.time.attrs['bounds'] = 'time_bound'
    return ds.copy(True)


def xr_ds_ex(decode_times=True, nyrs=3, var_const=True):
    """return an example xarray.Dataset object, useful for testing functions"""

    # set up values for Dataset, 4 yrs of analytic monthly values
    days_1yr = np.array([31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0])
    time_edges = np.insert(np.cumsum(npm.repmat(days_1yr, nyrs, 1)), 0, 0)
    time_bounds_vals = np.stack((time_edges[:-1], time_edges[1:]), axis=1)
    time_vals = np.mean(time_bounds_vals, axis=1)
    time_vals_yr = time_vals / 365.0
    if var_const:
        var_vals = np.ones_like(time_vals_yr)
    else:
        var_vals = np.sin(np.pi * time_vals_yr) * np.exp(-0.1 * time_vals_yr)

    time_units = 'days since 0001-01-01'
    calendar = 'noleap'

    if decode_times:
        time_vals = cftime.num2date(time_vals, time_units, calendar)
        time_bounds_vals = cftime.num2date(time_bounds_vals, time_units, calendar)

    # create Dataset, including time_bounds
    time_var = xr.DataArray(
        time_vals,
        name='time',
        dims='time',
        coords={'time': time_vals},
        attrs={'bounds': 'time_bounds'},
    )
    if not decode_times:
        time_var.attrs['units'] = time_units
        time_var.attrs['calendar'] = calendar
    time_bounds = xr.DataArray(
        time_bounds_vals, name='time_bounds', dims=('time', 'd2'), coords={'time': time_var}
    )
    var = xr.DataArray(var_vals, name='var_ex', dims='time', coords={'time': time_var})
    ds = var.to_dataset()
    ds = xr.merge((ds, time_bounds))

    if decode_times:
        ds.time.encoding['units'] = time_units
        ds.time.encoding['calendar'] = calendar

    return ds
