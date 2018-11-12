#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np


def _apply_nan_mask(da, weights, avg_over_dims_v):
    weights = weights.where(da.notnull())
    np.testing.assert_allclose(
        (weights / weights.sum(avg_over_dims_v)).sum(avg_over_dims_v), 1.0
    )
    return weights


def _get_op_over_dims(da, weights, dim):
    if not dim:
        dim = weights.dims

    op_over_dims_v = [k for k in dim if k in da.dims]
    return op_over_dims_v


def da_weighted_sum(da, weights, dim=None):

    sum_over_dims_v = _get_op_over_dims(da, weights, dim)
    if not sum_over_dims_v:
        raise ValueError(
            "Unexpected dimensions for variable {0}".format(
                da.name))

    da_output = (da * weights).sum(sum_over_dims_v)
    return da_output


def da_weighted_mean(da, weights, dim=None, apply_nan_mask=True):

    avg_over_dims_v = _get_op_over_dims(da, weights, dim)
    if not avg_over_dims_v:
        raise ValueError(
            (
                "Unexpected dimensions for variable {0}: {1}\n\n"
                "Average over dimensions: {2}"
            ).format(da.name, da, dim)
        )

    if apply_nan_mask:
        weights = _apply_nan_mask(da, weights, avg_over_dims_v)

    da_output = (da * weights).sum(avg_over_dims_v) / \
        weights.sum(avg_over_dims_v)
    return da_output


def da_weighted_std(da, weights, dim=None, apply_nan_mask=True, ddof=0):
    avg_over_dims_v = _get_op_over_dims(da, weights, dim)
    if not avg_over_dims_v:
        raise ValueError(
            (
                "Unexpected dimensions for variable {0}: {1}\n\n"
                "Average over dimensions: {2}"
            ).format(da.name, da, dim)
        )
    if apply_nan_mask:
        weights = _apply_nan_mask(da, weights, avg_over_dims_v)

    weighted_mean = da_weighted_mean(
        da, weights, dim=dim, apply_nan_mask=False)
    da_output = np.sqrt(
        (weights * (da - weighted_mean) ** 2).sum(avg_over_dims_v)
        / (weights.sum(avg_over_dims_v) - ddof)
    )
    return da_output
