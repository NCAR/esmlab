#!/usr/bin/env python
"""Top-level module for esmlab."""


import sys

from . import climatology, datasets, statistics
from ._version import get_versions
from .config import get_options, set_options

if sys.version_info > (3, 0):
    from .regrid import regridder

__version__ = get_versions()["version"]
del get_versions


_module_imports = (climatology, statistics, datasets, set_options, get_options)
__all__ = list(map(lambda x: x.__name__, _module_imports))
