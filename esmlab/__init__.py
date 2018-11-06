#!/usr/bin/env python
"""Top-level package for esmlab."""
from ._version import get_versions
from esmlab.accessors import EsmDataArrayAccessor

_module_imports = (EsmDataArrayAccessor,)
__version__ = get_versions()["version"]
del get_versions

__all__ = list(map(lambda x: x.__name__, _module_imports))
