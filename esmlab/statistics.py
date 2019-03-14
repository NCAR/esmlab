#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

from warnings import warn

import numpy as np
import xarray as xr

from .utils.common import esmlab_xr_set_options
from .utils.variables import (
    get_original_attrs,
    get_static_variables,
    get_variables,
    save_metadata,
    set_metadata,
    set_static_variables,
    update_attrs,
)


def weighted_mean_da(da, dim=None, weights=None):
    """ Compute weighted mean for DataArrays
    Parameters
    ----------
    da : xarray.DataArray
        DataArray for which to compute `weighted mean`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply mean.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of da.

    Returns
    -------
    reduced : DataArray
        New DataArray with mean applied to its data and the indicated
        dimension(s) removed.

    """
    if weights is None:
        warn('Computing mean using equal weights for all data points')
        return da.mean(dim)

    else:
        if not isinstance(weights, xr.DataArray):
            raise ValueError('Weights must be an xarray DataArray')
        # if NaN are present, we need to use individual weights
        if da.notnull().any():
            total_weights = weights.where(da.notnull()).sum(dim=dim)
        else:
            total_weights = weights.sum(dim)

        return (da * weights).sum(dim) / total_weights


def weighted_mean_ds(ds, dim=None, weights=None):
    """ Compute weighted mean for DataArrays
    Parameters
    ----------
    da : xarray.Dataset
        Dataset for which to compute `weighted mean`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply mean.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : Dataset
        New Dataset with mean applied to its data and the indicated
        dimension(s) removed.

    """
    if weights is None:
        warn('Computing mean using equal weights for all data points')
        return ds.mean(dim)
    else:
        ds.apply(weighted_mean_da, dim=dim, weights=weights)


@esmlab_xr_set_options(arithmetic_join='exact')
def weighted_mean(data, dim=None, weights=None):
    """ Compute weighted mean for xarray objects
    Parameters
    ----------
    data : Dataset or DataArray
         xarray object for which to compute `weighted mean`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply mean.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : Dataset or DataArray
        New xarray object with weighted mean applied to its data and the indicated
        dimension(s) removed.
    """
    if isinstance(data, xr.DataArray):
        return weighted_mean_da(data, dim, weights)
    elif isinstance(data, xr.Dataset):
        return weighted_mean_ds(data, dim, weights)
    else:
        raise ValueError('Data must be an xarray Dataset or DataArray')


def weighted_std_da(da, dim=None, weights=None, ddof=0):
    """ Compute weighted standard deviation for DataArrays
    Parameters
    ----------
    da : xarray.DataArray
        DataArray for which to compute `weighted std`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply standard deviation.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of da.
    ddof : int, optional
          Delta Degrees of Freedom. By default ddof is zero.

    Returns
    -------
    reduced : DataArray
        New DataArray with standard deviation applied to its data and the indicated
        dimension(s) removed.

    """
    if weights is None:
        warn('Computing standard deviation using equal weights for all data points')
        return da.std(dim)

    else:
        if not isinstance(weights, xr.DataArray):
            raise ValueError('Weights must be an xarray DataArray')
        # if NaN are present, we need to use individual weights
        if da.notnull().any():
            total_weights = weights.where(da.notnull()).sum(dim=dim)
        else:
            total_weights = weights.sum(dim)
        da_mean = weighted_mean_da(da, dim, weights)
        std = np.sqrt((weights * (da - da_mean) ** 2).sum(dim) / (total_weights - ddof))
        return std


def weighted_std_ds(ds, dim=None, weights=None):
    """ Compute weighted standard deviation for DataArrays
    Parameters
    ----------
    da : xarray.Dataset
        Dataset for which to compute `weighted standard deviation`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply standard deviation.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : Dataset
        New Dataset with standard deviation applied to its data and the indicated
        dimension(s) removed.

    """
    if weights is None:
        warn('Computing standard deviation using equal weights for all data points')
        return ds.std(dim)
    else:
        ds.apply(weighted_std_da, dim=dim, weights=weights)


@esmlab_xr_set_options(arithmetic_join='exact')
def weighted_std(data, dim=None, weights=None):
    """ Compute weighted standard deviation for xarray objects
    Parameters
    ----------
    data : Dataset or DataArray
         xarray object for which to compute `weighted std`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply standard deviation.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : Dataset or DataArray
        New xarray object with weighted standard deviation applied to its data and the indicated
        dimension(s) removed.
    """
    if isinstance(data, xr.DataArray):
        return weighted_std_da(data, dim, weights)
    elif isinstance(data, xr.Dataset):
        return weighted_std_ds(data, dim, weights)
    else:
        raise ValueError('Data must be an xarray Dataset or DataArray')


@esmlab_xr_set_options(arithmetic_join='exact')
def weighted_rmsd(x, y, dim=None, weights=None):
    """ Compute weighted root mean square deviation between two xarray Objects

    Parameters
    ----------
    x, y : xarray objects
        xarray objects (Dataset/DataArray) for which to compute `weighted_rmsd`.
    dim : str or sequence of str, optional
        Dimension(s) over which to apply rmsd.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : Dataset or DataArray
        New Dataset/DataArray with root mean square deviation applied to x, y and the indicated
        dimension(s) removed.
    """
    dev = (x - y) ** 2
    dev_mean = weighted_mean(dev, dim, weights)
    return np.sqrt(dev_mean)


@esmlab_xr_set_options(arithmetic_join='exact')
def weighted_cov(x, y, dim=None, weights=None):
    """ Compute weighted covariance between two xarray objects.

    Parameters
    ----------

    x, y : xarray objects
        xarray objects (Dataset/DataArray) for which to compute `weighted covariance`.

    dim : str or sequence of str, optional
        Dimension(s) over which to apply covariance.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : Dataset or DataArray
        New Dataset/DataArray with covariance applied to x, y and the indicated
        dimension(s) removed.
    """

    mean_x = weighted_mean(x, dim, weights)
    mean_y = weighted_mean(y, dim, weights)
    dev_x = x - mean_x
    dev_y = y - mean_y
    dev_xy = dev_x * dev_y
    cov_xy = weighted_mean(dev_xy, dim, weights)
    return cov_xy


def _apply_nan_mask(weights, x, y=None):
    # If y is specified, make sure x and y have same shape
    if y is not None and isinstance(y, xr.DataArray):
        assert x.shape == y.shape
        valid = x.notnull() & y.notnull()
    else:
        valid = x.notnull()

    # Apply nan mask
    return weights.where(valid)


def _get_weights_and_dims(x, y=None, weights=None, dim=None, apply_nan_mask=True):
    """ Get weights and dimensions """

    if dim and isinstance(dim, str):
        dims = [dim]

    elif isinstance(dim, list):
        dims = dim

    else:
        dims = [k for k in x.dims]

    op_over_dims = [k for k in dims if k in x.dims]
    if not op_over_dims:
        raise ValueError('Unexpected dimensions for variable {0}'.format(x.name))

    dims_shape = tuple(l for i, l in enumerate(x.shape) if x.dims[i] in op_over_dims)
    if weights is None:
        weights = xr.DataArray(np.ones(dims_shape), dims=op_over_dims)
        weights = _apply_nan_mask(weights, x, y)

    else:
        assert weights.shape == dims_shape
        if apply_nan_mask:
            weights = _apply_nan_mask(weights, x, y)

    # Make sure weights add up to 1.0
    rtol = 1e-6 if weights.dtype == np.float32 else 1e-7
    np.testing.assert_allclose(
        (weights / weights.sum(op_over_dims)).sum(op_over_dims), 1.0, rtol=rtol
    )
    return weights, op_over_dims


@esmlab_xr_set_options(arithmetic_join='exact')
def weighted_sum(x, weights=None, dim=None, apply_nan_mask=True):
    """Reduce `xarray.DataArray` by applying `weighted sum` along some dimension(s).

            Parameters
            ----------

            x : `xarray.DataArray`
               xarray object for which to compute `weighted sum`.

            weights : array_like, optional
                    weights to use. By default, weights=`None`

            dim : str or sequence of str, optional
                Dimension(s) over which to apply mean. By default `weighted sum`
                is applied over all dimensions.

            apply_nan_mask : bool, default: True

            Returns
            -------

            Weighted_sum : `xarray.DataArray`
                New DataArray object with `weighted sum` applied to its data
                and the indicated dimension(s) removed. If `weights` is None,
                returns regular sum using equal weights for all data points.
    """
    if weights is None:
        warn('Computing sum using equal weights for all data points')

    weights, op_over_dims = _get_weights_and_dims(
        x, weights=weights, dim=dim, apply_nan_mask=apply_nan_mask
    )
    x_w_sum = (x * weights).sum(op_over_dims)

    original_attrs, original_encoding = get_original_attrs(x)
    return update_attrs(x_w_sum, original_attrs, original_encoding)


@esmlab_xr_set_options(arithmetic_join='exact')
def weighted_corr(x, y, weights=None, dim=None, apply_nan_mask=True):
    """ Compute weighted correlation between two `xarray.DataArray` objects.

    Parameters
    ----------

    x, y : `xarray.DataArray` objects
        xarray objects for which to compute `weighted correlation`.

    weights : array_like, optional
             weights to use. By default, weights=`None`

    dim : str or sequence of str, optional
           Dimension(s) over which to apply `weighted correlation`
           By default weighted correlation is applied over all dimensions.

    apply_nan_mask : bool, default: True

    Returns
    -------

    weighted_correlation : float
              If `weights` is None, returns correlation using equal weights for all data points.


    """
    if weights is None:
        warn('Computing weighted correlation using equal weights for all data points')

    weights, op_over_dims = _get_weights_and_dims(
        x, weights=weights, dim=dim, apply_nan_mask=apply_nan_mask
    )

    # If the mask is applied in previous operation,
    # disable it for subseqent operations to speed up computation
    if apply_nan_mask:
        apply_nan_mask_flag = False
    else:
        apply_nan_mask_flag = True

    numerator = weighted_cov(
        x=x, y=y, weights=weights, dim=op_over_dims, apply_nan_mask=apply_nan_mask_flag
    )
    denominator = np.sqrt(
        weighted_cov(x, x, weights=weights, dim=op_over_dims, apply_nan_mask=apply_nan_mask_flag)
        * weighted_cov(y, y, weights=weights, dim=op_over_dims, apply_nan_mask=apply_nan_mask_flag)
    )
    corr_xy = numerator / denominator
    return corr_xy
