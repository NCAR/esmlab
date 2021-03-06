#!/usr/bin/env python
"""Top-level module for esmlab."""


from pkg_resources import DistributionNotFound, get_distribution

from . import config, datasets
from .common_utils import esmlab_xr_set_options
from .core import EsmlabAccessor, anomaly, climatology, resample
from .statistics import (
    mae,
    mse,
    rmse,
    weighted_corr,
    weighted_cov,
    weighted_mean,
    weighted_rmsd,
    weighted_std,
    weighted_sum,
)
from .utils.print_versions import show_versions

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
