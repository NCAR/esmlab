#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import os

import xarray as xr

from esmlab import config
from esmlab.datasets import open_dataset

_default_cache_dir = config.get('esmlab.sample_data_dir')


def test_open_dataset():
    ds = open_dataset('cesm_pop_yearly')
    assert isinstance(ds, xr.Dataset)


def test_open_dataset_cache():
    ds = open_dataset('ncep_forecast_tseries', cache=False)
    assert isinstance(ds, xr.Dataset)
    filepath = _default_cache_dir + '/ncep_forecast_tseries.nc'
    assert not os.path.exists(os.path.abspath(filepath))
