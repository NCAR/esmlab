#!/usr/bin/env python
from .config import set_options, get_options

"""Top-level package for esmlab."""
from ._version import get_versions
from . import climatology
from . import statistics

__version__ = get_versions()["version"]
del get_versions


_module_imports = (climatology, statistics)
__all__ = list(map(lambda x: x.__name__, _module_imports))
