#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import versioneer

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()


requirements = ["pandas>=0.23.0", "dask", "xarray", "netcdf4"]

test_requirements = ["pytest", "flake8"]

setup(
    maintainer="Anderson Banihirwe",
    maintainer_email="abanihi@ucar.edu",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
    ],
    description="",
    install_requires=requirements,
    license="https://github.com/NCAR/esmlab/blob/master/LICENSE.rst",
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords="esmlab",
    name="esmlab",
    packages=find_packages(),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/NCAR/esmlab",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
