#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import xarray as xr

from esmlab.datasets import open_dataset


def test_open_dataset():
    ds = open_dataset('cesm_cice_daily')
    assert isinstance(ds, xr.Dataset)
