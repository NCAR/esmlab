API reference
=============


.. currentmodule:: esmlab

Statistics functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   statistics.weighted_sum
   statistics.weighted_mean
   statistics.weighted_std
   statistics.weighted_rmsd
   statistics.weighted_cov
   statistics.weighted_corr

Climatology functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   climatology.compute_mon_climatology
   climatology.compute_mon_anomaly
   climatology.compute_ann_mean

Regridding
~~~~~~~~~~

.. autosummary::
   regridding.regridder

.. currentmodule:: esmlab.statistics

.. autofunction:: weighted_sum
.. autofunction:: weighted_mean
.. autofunction:: weighted_std
.. autofunction:: weighted_rmsd
.. autofunction:: weighted_cov
.. autofunction:: weighted_corr

.. currentmodule:: esmlab.climatology

.. autofunction:: compute_mon_climatology
.. autofunction:: compute_mon_anomaly
.. autofunction:: compute_ann_mean

.. currentmodule:: esmlab.regrid

.. autoclass:: regridder
   :members: __init__, regrid_dataarray
