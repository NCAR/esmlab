#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import versioneer
from os.path import exists

if exists("README.rst"):
    with open("README.rst") as f:
        long_description = f.read()
else:
    long_description = ""

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

test_requirements = ["pytest"]

setup(
    maintainer="Anderson Banihirwe",
    maintainer_email="abanihi@ucar.edu",
    description="Tools for working with earth system multi-model analyses with xarray",
    install_requires=install_requires,
    license="Apache License 2.0",
    long_description=long_description,
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
