#!/usr/bin/env python
"""Top-level module for esmlab."""


import sys

from pkg_resources import DistributionNotFound, get_distribution

from . import config, datasets
from .common_utils import esmlab_xr_set_options
from .core import EsmlabAccessor, anomaly, climatology, resample
from .statistics import (
    weighted_corr,
    weighted_cov,
    weighted_mean,
    weighted_rmsd,
    weighted_std,
    weighted_sum,
)

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
