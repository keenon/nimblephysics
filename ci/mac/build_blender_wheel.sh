#!/bin/bash
set -e

export VERSION=$(cat ../../VERSION.txt)
PYTHON=/Applications/Blender.app/Contents/Resources/3.3/python/bin/python3.10

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
$PYTHON setup.py install

# Actually push the wheel to PyPI
# python3 -m pip install --user --upgrade twine
# python3 -m twine upload --repository pypi wheelhouse/*