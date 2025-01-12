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
echo "python3=$(which python3)"

rm -rf dist/*
rm -rf build/*
# rm -rf wheelhouse/*

# Actually build the code
python3 setup.py sdist bdist_wheel

# Install delocate, to bundle dependencies into the wheel
cd dist
WHEEL_NAME=$(ls *.whl)
echo "WHEEL_NAME=${WHEEL_NAME}"
python3 -m auditwheel repair --no-update-tags --plat linux_aarch64 ${WHEEL_NAME}