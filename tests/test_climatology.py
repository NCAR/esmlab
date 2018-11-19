#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import xarray as xr
import numpy as np
import pytest
from esmlab.climatology import (
    compute_mon_climatology,
    compute_mon_anomaly,
    compute_ann_climatology)


def get_dataset_1():
    return xr.tutorial.open_dataset('rasm').load()


@pytest.mark.parametrize('dset', [get_dataset_1()])
def test_compute_mon_climatology(dset):
    computed_dset = compute_mon_climatology(dset)
    assert isinstance(computed_dset, xr.Dataset)


@pytest.mark.parametrize('dset', [get_dataset_1()])
def test_compute_mon_anomaly(dset):
    computed_dset = compute_mon_anomaly(dset)
    assert isinstance(computed_dset, xr.Dataset)


@pytest.mark.parametrize('dset', [get_dataset_1()])
def test_compute_ann_climatology(dset):
    computed_dset = compute_ann_climatology(dset)
    assert isinstance(computed_dset, xr.Dataset)
