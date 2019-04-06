#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

from warnings import warn

import dask.array as dask_array
import numpy as np
import xarray as xr
from scipy import special

from .common_utils import esmlab_xr_set_options


def validate_weights(da, dim, weights):

    if dim is None:
        dim = list(da.dims)

    if isinstance(weights, (list, np.ndarray, dask_array.Array)):
        raise ValueError(
            'weights must be an xarray.DataArray with shape that is broadcastable to shape= {da.data.shape} of da.'
        )
    # if NaN are present, we need to use individual weights
    total_weights = weights.where(da.notnull()).sum(dim=dim)

    # Make sure weights add up to 1.0
    rtol = 1e-6 if weights.dtype == np.float32 else 1e-7
    np.testing.assert_allclose((weights / weights.sum(dim)).sum(dim), 1.0, rtol=rtol)

    return weights, total_weights


def weighted_sum_da(da, dim=None, weights=None):

    """ Compute weighted mean for DataArray
    Parameters
    ----------
    da : xarray.DataArray
        DataArray for which to compute `weighted mean`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply mean.
    weights : xarray.DataArray or array-like
        weights to apply. Shape must be broadcastable to shape of da.

    Returns
    -------
    reduced : xarray.DataArray
        New DataArray with mean applied to its data and the indicated
        dimension(s) removed.

    """
    if weights is None:
        return da.sum(dim)
    else:
        weights, _ = validate_weights(da, dim, weights)
        return (da * weights).sum(dim)


@esmlab_xr_set_options(arithmetic_join='exact', keep_attrs=True)
def weighted_sum(data, dim=None, weights=None):
    """ Compute weighted sum for xarray objects

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
         xarray object for which to compute `weighted sum`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply sum.
    weights : xarray.DataArray or array-like
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : xarray.Dataset or xarray.DataArray
        New xarray object with weighted sum applied to its data and the indicated
        dimension(s) removed.
    """
    if isinstance(data, xr.DataArray):
        return weighted_sum_da(data, dim, weights)
    elif isinstance(data, xr.Dataset):
        return data.apply(weighted_sum_da, dim=dim, weights=weights)
    else:
        raise ValueError('Data must be an xarray Dataset or DataArray')


def weighted_mean_da(da, dim=None, weights=None):
    """ Compute weighted mean for DataArray

    Parameters
    ----------
    da : xarray.DataArray
        DataArray for which to compute `weighted mean`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply mean.
    weights : xarray.DataArray or array-like
        weights to apply. Shape must be broadcastable to shape of da.

    Returns
    -------
    reduced : xarray.DataArray
        New DataArray with mean applied to its data and the indicated
        dimension(s) removed.

    """
    if weights is None:
        return da.mean(dim)

    elif all(d in da.dims for d in dim):
        weights, total_weights = validate_weights(da, dim, weights)
        return (da * weights).sum(dim) / total_weights

    else:
        return da


@esmlab_xr_set_options(arithmetic_join='exact', keep_attrs=True)
def weighted_mean(data, dim=None, weights=None):
    """ Compute weighted mean for xarray objects

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
         xarray object for which to compute `weighted mean`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply mean.
    weights : xarray.DataArray or array-like
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : xarray.Dataset or xarray.DataArray
        New xarray object with weighted mean applied to its data and the indicated
        dimension(s) removed.
    """
    if isinstance(data, xr.DataArray):
        return weighted_mean_da(data, dim, weights)
    elif isinstance(data, xr.Dataset):
        return data.apply(weighted_mean_da, dim=dim, weights=weights)

    else:
        raise ValueError('Data must be an xarray Dataset or DataArray')


def weighted_std_da(da, dim=None, weights=None, ddof=0):
    """ Compute weighted standard deviation for DataArray

    Parameters
    ----------
    da : xarray.DataArray
        DataArray for which to compute `weighted std`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply standard deviation.
    weights : xarray.DataArray or array-like
        weights to apply. Shape must be broadcastable to shape of da.
    ddof : int, optional
          Delta Degrees of Freedom. By default ddof is zero.

    Returns
    -------
    reduced : xarray.DataArray
        New DataArray with standard deviation applied to its data and the indicated
        dimension(s) removed.

    """
    if weights is None:
        return da.std(dim)

    elif all(d in da.dims for d in dim):
        weights, total_weights = validate_weights(da, dim, weights)
        da_mean = weighted_mean_da(da, dim, weights)
        std = np.sqrt((weights * (da - da_mean) ** 2).sum(dim) / (total_weights - ddof))
        return std

    else:
        return da


@esmlab_xr_set_options(arithmetic_join='exact', keep_attrs=True)
def weighted_std(data, dim=None, weights=None):
    """ Compute weighted standard deviation for xarray objects

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
         xarray object for which to compute `weighted std`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply standard deviation.
    weights : xarray.DataArray or array-like
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : xarray.Dataset or xarray.DataArray
        New xarray object with weighted standard deviation applied to its data and the indicated
        dimension(s) removed.
    """
    if isinstance(data, xr.DataArray):
        return weighted_std_da(data, dim, weights)
    elif isinstance(data, xr.Dataset):
        return data.apply(weighted_std_da, dim=dim, weights=weights)
    else:
        raise ValueError('Data must be an xarray Dataset or DataArray')


@esmlab_xr_set_options(arithmetic_join='exact', keep_attrs=True)
def weighted_rmsd(x, y, dim=None, weights=None):
    """ Compute weighted root mean square deviation between two xarray DataArrays

    Parameters
    ----------
    x, y : xarray.DataArray
        xarray DataArray for which to compute `weighted_rmsd`.
    dim : str or sequence of str, optional
        Dimension(s) over which to apply rmsd.
    weights : xarray.DataArray or array-like
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : xarray.DataArray
        New DataArray with root mean square deviation applied to x, y and the indicated
        dimension(s) removed.
    """
    if not isinstance(x, xr.DataArray) or not isinstance(y, xr.DataArray):
        raise ValueError('x and y must be xarray DataArrays')
    dev = (x - y) ** 2
    dev_mean = weighted_mean(dev, dim, weights)
    return np.sqrt(dev_mean)


@esmlab_xr_set_options(arithmetic_join='exact', keep_attrs=True)
def weighted_cov(x, y, dim=None, weights=None):
    """ Compute weighted covariance between two xarray DataArrays.

    Parameters
    ----------

    x, y : xarray.DataArray
        xarray DataArrays for which to compute `weighted covariance`.

    dim : str or sequence of str, optional
        Dimension(s) over which to apply covariance.
    weights : xarray.DataArray or array-like
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : DataArray
        New DataArray with covariance applied to x, y and the indicated
        dimension(s) removed.
    """

    if not isinstance(x, xr.DataArray) or not isinstance(y, xr.DataArray):
        raise ValueError('x and y must be xarray DataArrays')
    mean_x = weighted_mean(x, dim, weights)
    mean_y = weighted_mean(y, dim, weights)
    dev_x = x - mean_x
    dev_y = y - mean_y
    dev_xy = dev_x * dev_y
    cov_xy = weighted_mean(dev_xy, dim, weights)
    return cov_xy


@esmlab_xr_set_options(arithmetic_join='exact')
def weighted_corr(x, y, dim=None, weights=None, return_p=True):
    """ Compute weighted correlation between two `xarray.DataArray`.

    Parameters
    ----------

    x, y : xarray.DataArray
        xarray DataArrays for which to compute `weighted correlation`.

    dim : str or sequence of str, optional
        Dimension(s) over which to apply correlation.
    weights : xarray.DataArray or array-like
        weights to apply. Shape must be broadcastable to shape of data.

    return_p : bool, default: True
        If True, compute and return the p-value(s) associated with the
        correlation.

    Returns
    -------
    reduced : xarray.DataArray
        New DataArray with correlation applied to x, y and the indicated
        dimension(s) removed.

        If `return_p` is True, appends the resulting p values to the
        returned Dataset.
    """

    if not isinstance(x, xr.DataArray) or not isinstance(y, xr.DataArray):
        raise ValueError('x and y must be xarray DataArrays')

    valid_values = x.notnull() & y.notnull()
    x = x.where(valid_values)
    y = y.where(valid_values)
    numerator = weighted_cov(x, y, dim, weights)
    denominator = np.sqrt(weighted_cov(x, x, dim, weights) * weighted_cov(y, y, dim, weights))
    corr_xy = numerator / denominator

    if return_p:
        p = compute_corr_significance(corr_xy, len(x))
        corr_xy.name = 'r'
        p.name = 'p'
        return xr.merge([corr_xy, p])
    else:
        return corr_xy


@esmlab_xr_set_options(arithmetic_join='exact', keep_attrs=True)
def compute_corr_significance(r, N):
    """ Compute statistical significance for a pearson correlation between
        two xarray objects.

    Parameters
    ----------
    r : `xarray.DataArray` object
        correlation coefficient between two time series.

    N : int
        length of time series being correlated.

    Returns
    -------
    pval : float
        p value for pearson correlation.

    """
    df = N - 2
    t_squared = r ** 2 * (df / ((1.0 - r) * (1.0 + r)))
    # method used in scipy, where `np.fmin` constrains values to be
    # below 1 due to errors in floating point arithmetic.
    pval = special.betainc(0.5 * df, 0.5, np.fmin(df / (df + t_squared), 1.0))
    return xr.DataArray(pval, coords=t_squared.coords, dims=t_squared.dims)
