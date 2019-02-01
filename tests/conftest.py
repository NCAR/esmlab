import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def dset():
    start_date = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    start_date = np.append(start_date, start_date + 365)
    end_date = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
    end_date = np.append(end_date, end_date + 365)
    ds = xr.Dataset(coords={"time": 24, "lat": 1, "lon": 1, "d2": 2})
    ds["time"] = xr.DataArray(end_date, dims="time")
    ds["lat"] = xr.DataArray([0], dims="lat")
    ds["lon"] = xr.DataArray([0], dims="lon")
    ds["d2"] = xr.DataArray([0, 1], dims="d2")
    ds["time_bound"] = xr.DataArray(
        np.array([start_date, end_date]).transpose(), dims=["time", "d2"]
    )
    ds["var_to_average"] = xr.DataArray(
        np.append(np.zeros([12, 1, 1]), np.ones([12, 1, 1]), axis=0),
        dims=["time", "lat", "lon"],
    )
    ds.time.attrs["units"] = "days since 0001-01-01 00:00:00"
    ds.time.attrs["calendar"] = "noleap"
    ds.time.attrs["bounds"] = "time_bound"
    return ds
