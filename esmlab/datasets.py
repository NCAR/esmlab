""" Useful for:
* Testing
* building tutorials in the documentation.
"""

from __future__ import absolute_import, division, print_function

import os

import xarray as xr

try:
    from urllib.request import urlretrieve

except ImportError:
    from urllib import urlretrieve


_default_cache_dir = os.sep.join(("~", ".esmlab_data"))


def open_dataset(
    name,
    cache=True,
    cache_dir=_default_cache_dir,
    github_url="https://github.com/NCAR/esmlab-data",
    branch="master",
    **kwargs
):
    """Load a dataset from the online repository (requires access to internet).

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str
        Name of the netcdf file containing the dataset
        ie. 'air_temperature'
    cache_dir : string, optional
        The directory in which to search for and write cached data.
    cache : boolean, optional
        If True, then cache data locally for use on subsequent calls
    github_url : string
        Github repository where the data is stored
    branch : string
        The git branch to download from
    kwargs : dict, optional
        Passed to xarray.open_dataset
    """
    longdir = os.path.expanduser(cache_dir)
    fullname = name + ".nc"
    localfile = os.sep.join((longdir, fullname))
    md5name = name + ".md5"
    md5file = os.sep.join((longdir, md5name))

    if not os.path.exists(localfile):

        # This will always leave this directory on disk.
        if not os.path.isdir(longdir):
            os.mkdir(longdir)

        url = "/".join((github_url, "raw", branch, fullname))
        urlretrieve(url, localfile)
        url = "/".join((github_url, "raw", branch, md5name))
        urlretrieve(url, md5file)

        localmd5 = xr.tutorial.file_md5_checksum(localfile)
        with open(md5file, "r") as f:
            remotemd5 = f.read()

        if localmd5 != remotemd5:
            os.remove(localfile)
            msg = """
            MD5 checksum does not match, try downloading dataset again.
            """
            raise IOError(msg)

    ds = xr.open_dataset(localfile, **kwargs)

    if not cache:
        ds = ds.load()
        os.remove(localfile)

    return ds
