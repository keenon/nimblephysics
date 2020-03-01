#!/usr/bin/env bash
set -ex

pacman -Syu --needed --noconfirm base-devel
pacman -Syu --needed --noconfirm cmake

if [ -z "$BUILD_DIR" ]; then
  echo "Error: Environment variable BUILD_DIR is unset. Using $PWD by default."
  BUILD_DIR=$PWD
fi

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
