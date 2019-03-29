#!/usr/bin/env python
"""Top-level module for esmlab."""


import sys

from . import datasets
from ._version import get_versions
from .core import EsmlabAccessor, mon_anomaly, mon_climatology, ann_mean
from .common_utils import esmlab_xr_set_options
from .statistics import (
    weighted_corr,
    weighted_mean,
    weighted_std,
    weighted_rmsd,
    weighted_cov,
    weighted_sum,
)

__version__ = get_versions()['version']
del get_versions
