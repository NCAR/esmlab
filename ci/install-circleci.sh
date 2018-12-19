#!/bin/bash

set -e
set -eo pipefail

export PYTHON_VERSION=$1
echo
echo "[install dependencies]"
conda create -y -q -n esmlab-dev -c conda-forge python=${PYTHON_VERSION}
conda env update -f environment-dev.yml
source activate esmlab-dev

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