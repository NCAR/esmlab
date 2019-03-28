#!/usr/bin/env python
"""Top-level module for esmlab."""


import sys

from . import climatology, datasets, statistics
from .core import EsmlabAccessor
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions


_module_imports = (climatology, statistics, datasets)
__all__ = list(map(lambda x: x.__name__, _module_imports))
