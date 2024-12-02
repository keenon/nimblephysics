#!/bin/bash
set -e

export VERSION=$(cat ../../VERSION.txt)
PYTHON=$(which python3)

# Find our python paths
export PYTHON_INCLUDE=$(python3-config --includes)
echo "PYTHON_INCLUDE=${PYTHON_INCLUDE}"
export PYTHON_LIB=$(python3-config --libs)
echo "PYTHON_LIB=${PYTHON_LIB}"

pushd ../..

rm -rf dist/*
rm -rf build/*
# rm -rf wheelhouse/*

# Actually build the code
$PYTHON setup.py sdist bdist_wheel

# Install delocate, to bundle dependencies into the wheel
pushd dist
WHEEL_NAME=$(ls *.whl)
echo "WHEEL_NAME=${WHEEL_NAME}"
DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:/usr/local/lib" delocate-wheel -w ../wheelhouse -v --dylibs-only ${WHEEL_NAME}
popd

# Actually push the wheel to PyPI
# python3 -m pip install --user --upgrade twine
python3 -m twine upload --repository pypi wheelhouse/*