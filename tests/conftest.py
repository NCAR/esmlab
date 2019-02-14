import sys

import numpy as np
import pytest
import xarray as xr

from esmlab.config import set_options

set_options(gridfile_directory="./tests/data")
