#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import xarray as xr

from esmlab.climatology import (
    compute_ann_mean,
    compute_mon_anomaly,
    compute_mon_climatology,
)


def test_compute_mon_climatology(dset):
    computed_dset = compute_mon_climatology(dset)
    np.testing.assert_equal(computed_dset.var_to_average.values, 0.5)


def test_compute_mon_anomaly(dset):
    computed_dset = compute_mon_anomaly(dset)
    assert isinstance(computed_dset, xr.Dataset)
    expected = np.array(
        [
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
    )
    np.testing.assert_equal(computed_dset.var_to_average.values.ravel(), expected)


def test_compute_ann_mean(dset):
    computed_dset = compute_ann_mean(dset)
    assert isinstance(computed_dset, xr.Dataset)
    expected = np.array([0.0, 1.0])
    np.testing.assert_equal(computed_dset.var_to_average.values.ravel(), expected)
