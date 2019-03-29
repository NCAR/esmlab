#!/usr/bin/env python
"""Top-level module for esmlab."""


import sys

from . import config, datasets
from ._version import get_versions
from .common_utils import esmlab_xr_set_options
from .core import EsmlabAccessor, ann_mean, mon_anomaly, mon_climatology
from .statistics import (
    weighted_corr,
    weighted_cov,
    weighted_mean,
    weighted_rmsd,
    weighted_std,
    weighted_sum,
)

__version__ = get_versions()['version']
del get_versions
