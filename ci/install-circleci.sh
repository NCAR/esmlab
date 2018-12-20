#!/bin/bash

set -e
set -eo pipefail

export PYTHON_VERSION=$1
export DEPLOY=$2


echo
echo "[install dependencies]"
conda create -y -q -n esmlab-dev -c conda-forge python=${PYTHON_VERSION}
conda env update -f environment-dev.yml
source activate esmlab-dev

if [ "${DEPLOY}" = "conda" ]; then
   echo "[Building and Deploying conda package]"
   conda install conda-build anaconda-client
   ./ci/upload-anaconda.sh
   return 0
fi

if [ "${DEPLOY}" = "pypi" ]; then
   echo "[Building and Deploying PyPI package]"
   python setup.py sdist
   pip wheel . -w dist
   return 0
fi

echo
echo "[install esmlab]"
pip install --no-deps -e .
conda list
echo "[finished install]"

echo
echo "[Running Tests]"
py.test --junitxml=test-reports/junit.xml --cov=./


echo
echo "[Upload coverage]"
# Upload coverage in each build, codecov.io merges the reports
codecov 


# Check documentation build only in one job
if [ "${PYTHON_VERSION}" = "3.6" ]; then
  echo 
  echo "[Building documentation]"
  pushd docs
  make html
  popd
fi

exit 0