from __future__ import absolute_import, division, print_function

import xarray as xr

try:
    from contextlib import ContextDecorator
except ImportError:
    from contextlib2 import ContextDecorator


class esmlab_xr_set_options(ContextDecorator):
    """ Enables a context manager to also be used as a decorator"""

    def __init__(self, **kwargs):
        self.old = xr.set_options(**kwargs).old

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        xr.set_options(**self.old)
        return
