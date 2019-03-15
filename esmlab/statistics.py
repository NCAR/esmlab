#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

from warnings import warn

import numpy as np
import xarray as xr

from .utils.common import esmlab_xr_set_options


def validate_weights(da, dim, weights):
    if not isinstance(weights, xr.DataArray):
        raise ValueError('Weights must be an xarray DataArray')
    # if NaN are present, we need to use individual weights
    if ~da.notnull().all():
        total_weights = weights.where(da.notnull()).sum(dim=dim)
    else:
        total_weights = weights.sum(dim)

    # Make sure weights add up to 1.0
    rtol = 1e-6 if weights.dtype == np.float32 else 1e-7
    np.testing.assert_allclose((weights / weights.sum(dim)).sum(dim), 1.0, rtol=rtol)

    return weights, total_weights


def weighted_sum_da(da, dim=None, weights=None):

    """ Compute weighted mean for DataArray
    Parameters
    ----------
    da : DataArray
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
        warn('Computing sum using equal weights for all data points')
        return da.sum(dim)
    else:
        weights, _ = validate_weights(da, dim, weights)
        return (da * weights).sum(dim)


def weighted_sum_ds(ds, dim=None, weights=None):
    """ Compute weighted sum for Dataset
    Parameters
    ----------
    da : Dataset
        Dataset for which to compute `weighted sum`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply sum.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : Dataset
        New Dataset with sum applied to its data and the indicated
        dimension(s) removed.

    """
    if weights is None:
        warn('Computing sum using equal weights for all data points')
        return ds.sum(dim)
    else:
        ds.apply(weighted_sum_da, dim=dim, weights=weights)


@esmlab_xr_set_options(arithmetic_join='exact', keep_attrs=True)
def weighted_sum(data, dim=None, weights=None):
    """ Compute weighted sum for xarray objects
    Parameters
    ----------
    data : Dataset or DataArray
         xarray object for which to compute `weighted sum`
    dim : str or sequence of str, optional
        Dimension(s) over which to apply sum.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : Dataset or DataArray
        New xarray object with weighted sum applied to its data and the indicated
        dimension(s) removed.
    """
    if isinstance(data, xr.DataArray):
        return weighted_sum_da(data, dim, weights)
    elif isinstance(data, xr.Dataset):
        return weighted_sum_ds(data, dim, weights)
    else:
        raise ValueError('Data must be an xarray Dataset or DataArray')


def weighted_mean_da(da, dim=None, weights=None):
    """ Compute weighted mean for DataArray
    Parameters
    ----------
    da : DataArray
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
        weights, total_weights = validate_weights(da, dim, weights)
        return (da * weights).sum(dim) / total_weights


def weighted_mean_ds(ds, dim=None, weights=None):
    """ Compute weighted mean for Dataset
    Parameters
    ----------
    da : Dataset
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


@esmlab_xr_set_options(arithmetic_join='exact', keep_attrs=True)
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
    """ Compute weighted standard deviation for DataArray
    Parameters
    ----------
    da : DataArray
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
        weights, total_weights = validate_weights(da, dim, weights)
        da_mean = weighted_mean_da(da, dim, weights)
        std = np.sqrt((weights * (da - da_mean) ** 2).sum(dim) / (total_weights - ddof))
        return std


def weighted_std_ds(ds, dim=None, weights=None):
    """ Compute weighted standard deviation for Dataset
    Parameters
    ----------
    da : Dataset
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


@esmlab_xr_set_options(arithmetic_join='exact', keep_attrs=True)
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


@esmlab_xr_set_options(arithmetic_join='exact', keep_attrs=True)
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


@esmlab_xr_set_options(arithmetic_join='exact', keep_attrs=True)
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


@esmlab_xr_set_options(arithmetic_join='exact', keep_attrs=True)
def weighted_corr(x, y, dim=None, weights=None):
    """ Compute weighted correlation between two xarray objects.

    Parameters
    ----------

    x, y : xarray objects
        xarray objects (Dataset/DataArray) for which to compute `weighted correlation`.

    dim : str or sequence of str, optional
        Dimension(s) over which to apply correlation.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : Dataset or DataArray
        New Dataset/DataArray with correlation applied to x, y and the indicated
        dimension(s) removed.

    """

    valid_values = x.notnull() & y.notnull()
    x = x.where(valid_values)
    y = y.where(valid_values)
    numerator = weighted_cov(x, y, dim, weights)
    denominator = np.sqrt(weighted_cov(x, x, dim, weights) * weighted_cov(y, y, dim, weights))
    corr_xy = numerator / denominator
    return corr_xy
