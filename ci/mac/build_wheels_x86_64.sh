#!/bin/bash
set -e

export VERSION=$(cat ../../VERSION.txt)

# Find our python paths
export PYTHON_INCLUDE=$(python3-config --includes)
echo "PYTHON_INCLUDE=${PYTHON_INCLUDE}"
export PYTHON_LIB=$(python3-config --libs)
echo "PYTHON_LIB=${PYTHON_LIB}"

pushd ../..

mkdir -p bin
ln -sfn $(which python3) ./bin/python
export PATH=$(pwd)/bin:${PATH}
echo "python=$(which python)"

rm -rf dist/*
rm -rf build/*
# rm -rf wheelhouse/*

# Actually build the code
python setup.py sdist bdist_wheel --plat-name macosx-11.7-x86_64

# Install delocate, to bundle dependencies into the wheel
pushd dist
WHEEL_NAME=$(ls *.whl)
echo "WHEEL_NAME=${WHEEL_NAME}"
DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:/usr/local/lib" delocate-wheel -w ../wheelhouse -v --dylibs-only ${WHEEL_NAME}
popd

# Replace the ABI tag with a more general version
# mv wheelhouse/diffdart-0.0.2-cp38-cp38-macosx_10_14_x86_64.whl wheelhouse/diffdart-0.0.2-cp38-abi3-macosx_10_14_x86_64.whl

# Actually push the wheel to PyPI
# python3 -m pip install --user --upgrade twine
# python3 -m twine upload --repository pypi wheelhouse/*