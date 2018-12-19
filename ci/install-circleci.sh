#!/bin/bash

set -e

echo
echo "[install dependencies]"
conda env update -f environment-dev.yml
source activate esmlab-dev
conda list

echo
echo "[install esmlab]"
pip install --no-deps -e .

echo "[finished install]"

exit 0