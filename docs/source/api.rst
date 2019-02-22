API reference
=============


.. currentmodule:: esmlab

.. autosummary::
   config.set_options
   config.get_options
   datasets.open_dataset

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
   regrid.regridder

Utilities
~~~~~~~~~~~

.. autosummary::
   utils.time.compute_time_var
   utils.time.uncompute_time_var



.. currentmodule:: esmlab.config
.. autoclass:: set_options
.. autofunction:: get_options

.. currentmodule:: esmlab.datasets
.. autofunction:: open_dataset




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
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. currentmodule:: esmlab.utils.time

.. autofunction:: compute_time_var
.. autofunction:: uncompute_time_var
