[![Build Status](https://travis-ci.org/NCAR/esmlab.svg?branch=master)](https://travis-ci.org/NCAR/esmlab)
[![codecov](https://codecov.io/gh/NCAR/esmlab/branch/master/graph/badge.svg)](https://codecov.io/gh/NCAR/esmlab)
[![Documentation Status](https://readthedocs.org/projects/esmlab/badge/?version=latest)](https://esmlab.readthedocs.io/en/latest/?badge=latest)
# esmlab (Earth System Model Lab)

Tools for working with earth system models outputs with xarray. `esmlab` extends `xarray` by providing convenient functionalities such as meta data propagation when doing computation on `xarray` objects. 


See the [documentation](https://esmlab.readthedocs.io/en/latest/) for more.


## Usage 

```python
# Import packages

In [1]: import xarray as xr

In [2]: import numpy as np

In [3]: import esmlab

# Create xarray data arrays

In [4]: maskedarea = xr.DataArray(np.ones((10, 10)), dims=("x", "y"))

In [5]: x = xr.DataArray(
   ...: np.random.uniform(
   ...:     0, 10, (10, 10)), dims=("x", "y"), name="x_var")
   ...: 

In [6]: x
Out[6]: 
<xarray.DataArray 'x_var' (x: 10, y: 10)>
array([[6.464174, 7.019343, 9.160343, 3.053262, 9.368475, 9.705359, 4.53911 ,
        4.32455 , 9.726268, 8.026548],
       [1.781286, 2.10246 , 7.111004, 5.193137, 7.199186, 7.133222, 3.526382,
        8.284735, 6.226455, 1.221961],
       [9.965352, 6.140129, 9.41478 , 1.013669, 0.422011, 2.805869, 0.799003,
        4.524927, 4.364747, 5.231814],
       [9.272777, 1.314724, 7.180548, 5.397058, 8.584827, 6.094685, 9.705401,
        9.820953, 8.669493, 1.814462],
       [6.755809, 0.708355, 2.492557, 9.58748 , 4.547628, 1.919822, 7.422673,
        3.845401, 2.180653, 0.905171],
       [6.897773, 9.802236, 3.397387, 0.644367, 3.102339, 2.27512 , 4.708471,
        2.497213, 9.04623 , 7.453865],
       [9.008711, 8.460758, 4.817923, 3.91315 , 1.894857, 7.039858, 0.417558,
        3.903161, 3.76364 , 2.970794],
       [9.275888, 9.071674, 4.236316, 8.292264, 3.317507, 6.773342, 2.309864,
        3.11896 , 1.432776, 2.987903],
       [8.166222, 2.845253, 2.533549, 5.497737, 4.803681, 6.870216, 8.580527,
        2.506437, 4.004796, 6.811375],
       [2.427979, 2.330199, 5.321273, 8.563104, 7.767206, 4.66075 , 1.192411,
        0.459279, 2.443299, 9.174439]])
Dimensions without coordinates: x, y

In [7]: x.attrs
Out[7]: OrderedDict()

In [8]: y = xr.DataArray(
   ...:     np.random.uniform(
   ...:         0, 10, (10, 10)), dims=("x", "y"), name="y_var")
   ...: 

In [9]: y[3, 3:10] = np.nan

In [10]: y
Out[10]: 
<xarray.DataArray 'y_var' (x: 10, y: 10)>
array([[5.740615, 9.126203, 8.736934, 5.102134, 4.0002  , 3.591627, 5.124793,
        5.765631, 8.1106  , 7.513179],
       [1.937614, 2.835284, 0.660109, 1.084765, 3.649332, 9.733115, 4.190793,
        7.509018, 4.473454, 6.397932],
       [1.362011, 8.444731, 9.576225, 0.695174, 9.863594, 2.263395, 6.49631 ,
        5.735136, 7.488803, 8.453338],
       [0.442691, 2.40794 , 9.883274,      nan,      nan,      nan,      nan,
             nan,      nan,      nan],
       [3.669326, 1.177924, 1.259322, 6.023701, 3.361583, 3.030204, 1.378297,
        6.050141, 3.921537, 3.152998],
       [5.331043, 5.010629, 0.355016, 6.615643, 3.019434, 9.439884, 1.85413 ,
        5.458737, 8.407518, 6.959354],
       [5.883195, 7.457091, 6.97772 , 9.446799, 1.45491 , 0.084423, 6.859892,
        8.53863 , 1.175537, 2.426813],
       [9.36148 , 5.312882, 4.125667, 4.397538, 9.675557, 5.139773, 5.573218,
        8.50347 , 7.03912 , 4.282664],
       [9.442249, 6.319131, 9.851219, 3.224227, 9.687352, 3.993602, 0.289464,
        3.772107, 2.16541 , 5.903818],
       [5.887596, 0.466224, 8.499517, 4.190959, 7.931901, 7.315625, 7.804534,
        9.79462 , 7.598355, 0.714479]])
Dimensions without coordinates: x, y

In [11]: y.attrs
Out[11]: OrderedDict()

# Now fill in some metadata

In [12]: x.attrs['units'] = 'meters'

In [13]: x.attrs['created on'] = '2018-01-01'

In [14]: y.attrs['units'] = 'celsius'

In [15]: y.attrs['created by'] = 'foo'

# Compute weighted mean for each xarray data array using `esm` accessor from esmlab

In [16]: x.esm.weighted_mean(weights=maskedarea)
Out[16]: 
<xarray.DataArray ()>
array(5.178617)
Attributes:
    units:       meters
    created on:  2018-01-01

In [17]: y.esm.weighted_mean(weights=maskedarea)
Out[17]: 
<xarray.DataArray ()>
array(5.295066)
Attributes:
    units:       celsius
    created by:  foo
```