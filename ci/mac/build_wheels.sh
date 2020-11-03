#!/bin/bash
set -e

VERSION=$(cat ../../VERSION.txt)

# Find our python paths
PYTHON_INCLUDE=$(python-config --includes)
echo "PYTHON_INCLUDE=${PYTHON_INCLUDE}"
PYTHON_LIB=$(python-config --libs)
echo "PYTHON_LIB=${PYTHON_LIB}"

pushd ../..

rm -rf dist/*
rm -rf build/*
# rm -rf wheelhouse/*

# Actually build the code
python3 setup.py sdist bdist_wheel

# Install delocate, to bundle dependencies into the wheel
pushd dist
WHEEL_NAME=$(ls *.whl)
echo "WHEEL_NAME=${WHEEL_NAME}"
DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:/usr/local/lib" delocate-wheel -w ../wheelhouse -v ${WHEEL_NAME}
popd

# Replace the ABI tag with a more general version
# mv wheelhouse/diffdart-0.0.2-cp38-cp38-macosx_10_14_x86_64.whl wheelhouse/diffdart-0.0.2-cp38-abi3-macosx_10_14_x86_64.whl

# Actually push the wheel to PyPI
# python3 -m pip install --user --upgrade twine
# python3 -m twine upload --repository pypi wheelhouse/*