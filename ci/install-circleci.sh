#!/bin/bash

set -e
set -eo pipefail

export PYTHON_VERSION=$1
echo
echo "[install dependencies]"
conda create -y -q -n esmlab-dev python=${PYTHON_VERSION}
conda env update -f environment-dev.yml
source activate esmlab-dev

echo
echo "[install esmlab]"
pip install --no-deps -e .
conda list
echo "[finished install]"

exit 0