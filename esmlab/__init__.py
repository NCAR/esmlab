#!/usr/bin/env python
"""Top-level package for esmlab."""

import sys
from .config import set_options, get_options

from ._version import get_versions
from . import climatology
from . import statistics

if sys.version_info > (3, 0):
    from .regrid import regridder

__version__ = get_versions()["version"]
del get_versions


_module_imports = (climatology, statistics)
__all__ = list(map(lambda x: x.__name__, _module_imports))
