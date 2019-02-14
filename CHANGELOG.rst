=================
Changelog history
=================

Esmlab v2019.2.1 (2019-02-12)
==============================

- Add time_boud to output of compute_ann_mean (:pr:`51`) `Matthew Long`_
- Add xarray alignment option to prevent using mismatching coordinates (:pr:`54`) `Anderson Banihirwe`_
- Add regridding functionality (:pr:`56`) `Matthew Long`_
- Handle time_bound on data read with decode_times=True (:pr:`59`) `Matthew Long`_
- Add interface to esmlab-data (:pr:`61`) `Anderson Banihirwe`_


Esmlab v2019.2.0 (2019-02-02)
==============================

- Rename compute_ann_climatology to compute_ann_mean (:pr:`33`) `Anderson Banihirwe`_
- Don't add NaNs for _FillValue (:pr:`34`) `Anderson Banihirwe`_
- Change time handling for compute_mon_climatology and compute_ann_mean (:pr:`37`) `Matthew Long`_
- Add slice_mon_clim_time argument to compute_mon_climatology (:pr:`37`) `Matthew Long`_
- Drop time_bound var from compute_ann_mean (:pr:`43`) `Matthew Long`_




.. _`Anderson Banihirwe`: https://github.com/andersy005
.. _`Matthew Long`: https://github.com/matt-long