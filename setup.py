#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import versioneer
import io
from os.path import exists

readme = io.open("README.rst").read() if exists("README.rst") else " "


requirements = ["pandas>=0.23.0", "dask", "xarray", "netcdf4", "cftime"]

test_requirements = ["pytest"]

setup(
    maintainer="Anderson Banihirwe",
    maintainer_email="abanihi@ucar.edu",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache-2.0",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
    ],
    description=readme,
    install_requires=requirements,
    license="Apache License 2.0",
    long_description=readme,
    keywords="esmlab xarray cmip",
    name="esmlab",
    packages=find_packages(),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/NCAR/esmlab",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
