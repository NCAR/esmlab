#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import numpy as np

from .utils import (
    get_grid_vars,
    get_original_attrs,
    get_variables,
    save_metadata,
    set_grid_vars,
    set_metadata,
    time_bound_var,
    update_attrs,
)


def _apply_nan_mask(x, weights, avg_over_dims_v):
    weights = weights.where(x.notnull())
    np.testing.assert_allclose(
        (weights / weights.sum(avg_over_dims_v)).sum(avg_over_dims_v), 1.0
    )
    return weights


def _apply_nan_mask_for_two_arrays(x, y, weights):
    valid = x.notnull() & y.notnull()
    weights = weights.where(valid)
    return weights


def _get_op_over_dims(x, weights, dim):
    if not dim:
        dim = weights.dims

    op_over_dims_v = [k for k in dim if k in x.dims]
    return op_over_dims_v


def weighted_sum(x, weights, dim=None):
    """Reduce DataArray by applying `weighted sum` along some dimension(s).

            Parameters
            ----------

            x : DataArray object
               xarray object for which to compute `weighted sum`.

            weights : array_like

            dim : str or sequence of str, optional
                Dimension(s) over which to apply mean. By default `weighted sum`
                is applied over all dimensions.

            Returns
            -------

            reduced : DataArray
                New DataArray object with `weighted sum` applied to its data
                and the indicated dimension(s) removed.
        """

    sum_over_dims_v = _get_op_over_dims(x, weights, dim)
    if not sum_over_dims_v:
        raise ValueError("Unexpected dimensions for variable {0}".format(x.name))

    x_w_sum = (x * weights).sum(sum_over_dims_v)
    original_attrs, original_encoding = get_original_attrs(x)
    return update_attrs(x_w_sum, original_attrs, original_encoding)


def weighted_mean(x, weights, dim=None, apply_nan_mask=True):
    """Reduce DataArray by applying weighted mean along some dimension(s).

        Parameters
        ----------

        x : DataArray object
           xarray object for which to compute `weighted mean`.

        weights : array_like

        dim : str or sequence of str, optional
           Dimension(s) over which to apply `weighted mean`. By default weighted mean
           is applied over all dimensions.

        apply_nan_mask : bool, default: True

        Returns
        -------

        reduced : DataArray
             New DataArray object with ` weighted mean` applied to its data
             and the indicated dimension(s) removed.
        """

    avg_over_dims_v = _get_op_over_dims(x, weights, dim)
    if not avg_over_dims_v:
        raise ValueError(
            (
                "Unexpected dimensions for variable {0}: {1}\n\n"
                "Average over dimensions: {2}"
            ).format(x.name, x, dim)
        )

    if apply_nan_mask:
        weights = _apply_nan_mask(x, weights, avg_over_dims_v)

    x_w_mean = (x * weights).sum(avg_over_dims_v) / weights.sum(avg_over_dims_v)
    original_attrs, original_encoding = get_original_attrs(x)
    return update_attrs(x_w_mean, original_attrs, original_encoding)


def weighted_std(x, weights, dim=None, apply_nan_mask=True, ddof=0):
    """Reduce DataArray by applying `weighted std` along some dimension(s).

        Parameters
        ----------

        x : DataArray object
           xarray object for which to compute `weighted std`.

        weights : array_like

        dim : str or sequence of str, optional
           Dimension(s) over which to apply mean. By default `weighted std`
           is applied over all dimensions.

        apply_nan_mask : bool, default: True

        ddof : int


        Returns
        -------

        reduced : DataArray
             New DataArray object with `weighted std` applied to its data
             and the indicated dimension(s) removed.
        """

    avg_over_dims_v = _get_op_over_dims(x, weights, dim)
    if not avg_over_dims_v:
        raise ValueError(
            (
                "Unexpected dimensions for variable {0}: {1}\n\n"
                "Average over dimensions: {2}"
            ).format(x.name, x, dim)
        )
    if apply_nan_mask:
        weights = _apply_nan_mask(x, weights, avg_over_dims_v)

    x_w_mean = weighted_mean(x, weights, dim=dim, apply_nan_mask=False)

    x_w_std = np.sqrt(
        (weights * (x - x_w_mean) ** 2).sum(avg_over_dims_v)
        / (weights.sum(avg_over_dims_v) - ddof)
    )
    original_attrs, original_encoding = get_original_attrs(x)

    return update_attrs(x_w_std, original_attrs, original_encoding)


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
    dev_mean = weighted_mean(dev, weights=weights, dim=dim, apply_nan_mask=False)
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

    mean_x = weighted_mean(x, weights=weights, dim=dim, apply_nan_mask=False)
    mean_y = weighted_mean(y, weights=weights, dim=dim, apply_nan_mask=False)

    dev_x = x - mean_x
    dev_y = y - mean_y
    dev_xy = dev_x * dev_y
    cov_xy = weighted_mean(dev_xy, weights=weights, dim=dim, apply_nan_mask=False)
    return cov_xy


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
    corr_xy = numerator / denominator
    return corr_xy
