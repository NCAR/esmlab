#!/usr/bin/env python

import os

import numpy as np
import pytest
import xarray as xr

from esmlab.regrid import regridder

_here = os.path.abspath(os.path.dirname(__file__))


def test_regrid_init():

    R = regridder(
        name_grid_src="T62",
        name_grid_dst="CAM_f09",
        method="bilinear",
        overwrite_existing=True,
    )

    assert isinstance(R, regridder)


def test_regrid_regrid():

    R = regridder(
        name_grid_src="T62",
        name_grid_dst="CAM_f09",
        method="bilinear",
        overwrite_existing=False,
    )

    ds = xr.open_dataset(_here + "/data/ncep.t_10.T62.2009.time0.nc")
    dao = R(ds.t_10)
    print(dao)
    assert isinstance(dao, xr.DataArray)
