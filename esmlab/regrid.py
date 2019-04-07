# flake8: noqa
try:
    from esmlab_regrid import *
except ImportError:
    msg = (
        'Esmlab-regrid is not installed.\n\n'
        'Please either conda or pip install esmlab-regrid:\n\n'
        '  conda install esmlab-regrid          # either conda install\n'
        '  pip install esmlab-regrid --upgrade  # or pip install'
    )
    raise ImportError(msg)
