#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from os.path import exists

from setuptools import find_packages, setup

if exists('README.rst'):
    with open('README.rst') as f:
        long_description = f.read()
else:
    long_description = ''


with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

test_requirements = ['pytest']
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering',
]


setup(
    maintainer='Anderson Banihirwe',
    maintainer_email='abanihi@ucar.edu',
    description='Tools for working with earth system multi-model analyses with xarray',
    install_requires=install_requires,
    python_requires='>3.5',
    license='Apache License 2.0',
    long_description=long_description,
    classifiers=CLASSIFIERS,
    name='esmlab',
    packages=find_packages(exclude=('tests',)),
    test_suite='tests',
    include_package_data=True,
    tests_require=test_requirements,
    url='https://github.com/NCAR/esmlab',
    use_scm_version={'version_scheme': 'post-release', 'local_scheme': 'dirty-tag'},
    setup_requires=['setuptools_scm', 'setuptools>=30.3.0', 'setuptools_scm_git_archive'],
    zip_safe=False,
)
