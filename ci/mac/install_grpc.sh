#!/bin/bash

set -e

# Install protobuf
PROTOBUF_VERSION="29.2"
# wget https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz
# tar -xvzf protobuf-all-${PROTOBUF_VERSION}.tar.gz
# rm protobuf-all-${PROTOBUF_VERSION}.tar.gz
# pushd protobuf-${PROTOBUF_VERSION}
# CXX_FLAGS="-fvisibility=hidden" ./configure
# make -j
# make install
# popd
# rm -rf protobuf-${PROTOBUF_VERSION}

# brew install zlib
brew install xz zlib bzip2

# Install grpc
git clone --recurse-submodules -b v1.69.0 https://github.com/grpc/grpc
pushd grpc
pushd third_party/protobuf
git checkout v${PROTOBUF_VERSION}
popd
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DgRPC_ZLIB_PROVIDER=package \
      -DCMAKE_OSX_DEPLOYMENT_TARGET="10.15" \
      -DCMAKE_CXX_FLAGS="-fvisibility=hidden" \
      $CMAKE_FLAGS \
      ../..
sudo make install -j
popd
popd
sudo rm -rf grpc