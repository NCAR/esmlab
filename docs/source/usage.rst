============
Usage
============


Import packages

.. ipython:: python

    import xarray as xr
    import numpy as np
    from esmlab import statistics as stats


Create ``xarray`` data arrays

.. ipython:: python

    maskedarea = xr.DataArray(np.ones((10, 10)), dims=("x", "y"))
    x = xr.DataArray(
    np.random.uniform(
        0, 10, (10, 10)), dims=("x", "y"), name="x_var")
    x
    x.attrs
    y = xr.DataArray(
        np.random.uniform(
            0, 10, (10, 10)), dims=("x", "y"), name="y_var")
    
    y[3, 3:10] = np.nan
    y
    y.attrs

Now fill in some metadata 

.. ipython:: python

    x.attrs['units'] = 'meters'
    x.attrs['created on'] = '2018-01-01'
    y.attrs['units'] = 'celsius'
    y.attrs['created by'] = 'foo'


.. ipython:: python

    x1 = stats.weighted_mean(x, weights=maskedarea)
    y1 = stats.weighted_mean(y, weights=maskedarea)
    print(x1)
    print(y1)
    assert x1.attrs == x.attrs
    assert x1.encoding == x.encoding


    
