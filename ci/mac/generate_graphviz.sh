#!/bin/bash
set -e

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

mkdir build_viz
pushd build_viz
cmake .. --graphviz=viz.dot
