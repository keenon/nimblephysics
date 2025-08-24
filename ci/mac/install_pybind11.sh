#!/bin/bash
set -e

# Install pybind11
git clone https://github.com/pybind/pybind11.git
pushd pybind11
git checkout v2.11.1
mkdir build
pushd build
cmake .. -DPYTHON_EXECUTABLE:FILEPATH=$(which python3.11) $CMAKE_FLAGS
sudo make install -j
popd
popd
sudo rm -rf pybind11