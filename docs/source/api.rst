API
===

.. currentmodule:: esmlab

EsmDataArrayAccessor Methods
""""""""""""""""""""""""""""
These methods can be accessed by using `esm` accessor on any 
`xarray DataArray` after importing `esmlab`.

.. autosummary::
   EsmDataArrayAccessor.weighted_sum
   EsmDataArrayAccessor.weighted_mean
   EsmDataArrayAccessor.weighted_std


Statistics functions
"""""""""""""""""""""
.. autosummary::
   statistics.weighted_rmsd
   statistics.weighted_cov
   statistics.weighted_corr 


.. autoclass:: EsmDataArrayAccessor
   :members:



.. currentmodule:: esmlab.statistics

.. autofunction:: weighted_rmsd
.. autofunction:: weighted_cov
.. autofunction:: weighted_corr