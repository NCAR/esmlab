#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
from .accessors import EsmDataArrayAccessor


def _apply_nan_mask_for_two_arrays(x, y, weights):
    valid = x.notnull() & y.notnull()
    weights = weights.where(valid)
    return weights


def weighted_rmsd(x, y, weights, dim=None):
    """ Compute weighted root-mean-square-deviation between two `xarray` DataArrays.

    Parameters
    ----------
    x, y : DataArray objects
        xarray objects for which to compute `weighted_rmsd`.

    weights : array_like

    dim : str or sequence of str, optional
           Dimension(s) over which to apply `weighted rmsd` By default weighted rmsd
           is applied over all dimensions.

    Returns
    -------
    root mean square deviation : float
    """

    if not dim:
        dim = weights.dims

    weights = _apply_nan_mask_for_two_arrays(x, y, weights)
    dev = (x - y) ** 2
    dev_mean = dev.esm.weighted_mean(weights=weights, dim=dim, apply_nan_mask=False)
    return np.sqrt(dev_mean)


def weighted_cov(x, y, weights, dim=None):
    """ Compute weighted covariance between two `xarray` DataArrays.

    Parameters
    ----------
    x, y : DataArray objects
        xarray objects for which to compute `weighted covariance`.

    weights : array_like

    dim : str or sequence of str, optional
           Dimension(s) over which to apply `weighted covariance`
           By default weighted covariance is applied over all dimensions.

    Returns
    -------
    covariance : float
    """

    if not dim:
        dim = weights.dims

    weights = _apply_nan_mask_for_two_arrays(x, y, weights)

    mean_x = x.esm.weighted_mean(weights=weights, dim=dim, apply_nan_mask=False)
    mean_y = y.esm.weighted_mean(weights=weights, dim=dim, apply_nan_mask=False)

    dev_x = x - mean_x
    dev_y = y - mean_y
    output = (dev_x * dev_y).esm.weighted_mean(
        weights=weights, dim=dim, apply_nan_mask=False
    )
    return output


def weighted_corr(x, y, weights, dim=None):
    """ Compute weighted correlation between two `xarray` DataArrays.

    Parameters
    ----------
    x, y : DataArray objects
        xarray objects for which to compute `weighted correlation`.

    weights : array_like

    dim : str or sequence of str, optional
           Dimension(s) over which to apply `weighted correlation`
           By default weighted correlation is applied over all dimensions.

    Returns
    -------
    correlation : float
    """

    numerator = weighted_cov(x, y, weights, dim)
    denominator = np.sqrt(
        weighted_cov(x, x, weights, dim) * weighted_cov(y, y, weights, dim)
    )
    output = numerator / denominator
    return output
