#!/usr/bin/env bash
set -ex

cd $BUILD_DIR/cpp14/dart
mkdir build && cd build
cmake ..
make -j4
make install

# Build an example using installed DART
cd $BUILD_DIR/cpp14/gazebo
mkdir build && cd build
cmake ..
make -j4
