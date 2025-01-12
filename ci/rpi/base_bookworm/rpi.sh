#!/bin/bash

set -euxo pipefail

# -------------------------------------------------------
# Update the package index and install system dependencies
# -------------------------------------------------------
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    yasm \
    m4 \
    libgmp3-dev \
    libblas-dev \
    liblapack-dev \
    libboost-all-dev \
    pkg-config \
    ca-certificates \
    gfortran \
    clang \
    lldb \
    lld

# -------------------------------------------------------
# Helper function to clean up temporary files
# -------------------------------------------------------
cleanup() {
    rm -rf "$@"
}

# -------------------------------------------------------
# Install Eigen
# -------------------------------------------------------
curl -L https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz -o eigen.tar.gz
tar -zxf eigen.tar.gz
cd eigen-3.3.7
mkdir build && cd build
cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j$(nproc)
sudo make install
cd ../..
cleanup eigen-3.3.7 eigen.tar.gz

# -------------------------------------------------------
# Install CCD
# -------------------------------------------------------
git clone https://github.com/danfis/libccd.git
cd libccd
mkdir build && cd build
cmake .. -DENABLE_DOUBLE_PRECISION=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j$(nproc)
sudo make install
cd ../..
cleanup libccd

# -------------------------------------------------------
# Install ASSIMP
# -------------------------------------------------------
git clone https://github.com/assimp/assimp.git
cd assimp
git checkout v5.0.1
mkdir build && cd build
cmake .. -DASSIMP_BUILD_ZLIB=ON -DASSIMP_BUILD_TESTS=ON -DASSIMP_BUILD_ASSIMP_TOOLS=OFF
make -j$(nproc)
sudo make install
cd ../..
cleanup assimp

# -------------------------------------------------------
# Install MUMPS
# -------------------------------------------------------
git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git
cd ThirdParty-Mumps
./get.Mumps
./configure
make -j$(nproc)
sudo make install
cd ..
cleanup ThirdParty-Mumps

# -------------------------------------------------------
# Install IPOPT
# -------------------------------------------------------
git clone https://github.com/coin-or/Ipopt.git
cd Ipopt
./configure --with-mumps
make -j$(nproc)
sudo make install
cd ..
cleanup Ipopt

# -------------------------------------------------------
# Install FCL
# -------------------------------------------------------
git clone https://github.com/flexible-collision-library/fcl.git
cd fcl
git checkout 0.3.4
mkdir build && cd build
cmake .. -DFCL_WITH_OCTOMAP=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j$(nproc)
sudo make install
cd ../..
cleanup fcl

# -------------------------------------------------------
# Install octomap
# -------------------------------------------------------
git clone https://github.com/OctoMap/octomap.git
cd octomap
git checkout v1.10.0
mkdir build && cd build
cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make
sudo make install
cd ../..
cleanup octomap

# -------------------------------------------------------
# Install tinyxml2
# -------------------------------------------------------
git clone https://github.com/leethomason/tinyxml2.git
cd tinyxml2
mkdir build && cd build
cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j$(nproc)
sudo make install
cd ../..
cleanup tinyxml2

# -------------------------------------------------------
# Install tinyxml
# -------------------------------------------------------
git clone https://github.com/robotology-dependencies/tinyxml.git
cd tinyxml
mkdir build && cd build
cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j$(nproc)
sudo make install
cd ../..
cleanup tinyxml

# -------------------------------------------------------
# Install urdfdom_headers
# -------------------------------------------------------
git clone https://github.com/ros/urdfdom_headers.git
cd urdfdom_headers
mkdir build && cd build
cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j$(nproc)
sudo make install
cd ../..
cleanup urdfdom_headers

# -------------------------------------------------------
# Install console_bridge
# -------------------------------------------------------
git clone https://github.com/ros/console_bridge.git
cd console_bridge
mkdir build && cd build
cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j$(nproc)
sudo make install
cd ../..
cleanup console_bridge

# -------------------------------------------------------
# Install urdfdom
# -------------------------------------------------------
git clone https://github.com/ros/urdfdom.git
cd urdfdom
mkdir build && cd build
cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j$(nproc)
sudo make install
cd ../..
cleanup urdfdom

# -------------------------------------------------------
# Install protobuf and gRPC
# -------------------------------------------------------
PROTOBUF_VERSION="29.2"
git clone --recurse-submodules -b v1.69.0 https://github.com/grpc/grpc
cd grpc/third_party/protobuf
git checkout v${PROTOBUF_VERSION}
cd ../..
mkdir cmake/build && cd cmake/build
cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON ../..
make -j$(nproc)
sudo make install
cd ../../..
cleanup grpc

# -------------------------------------------------------
# Install Google benchmark
# -------------------------------------------------------
git clone https://github.com/google/benchmark.git
git clone https://github.com/google/googletest.git benchmark/googletest
cd benchmark
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
make
sudo make install
cd ../..
cleanup benchmark

# -------------------------------------------------------
# Install MPFR
# -------------------------------------------------------
curl -L https://www.mpfr.org/mpfr-current/mpfr-4.2.1.tar.gz -o mpfr.tar.gz
tar -zxf mpfr.tar.gz
cd mpfr-4.2.1
./configure
make -j$(nproc)
sudo make install
cd ..
cleanup mpfr-4.2.1 mpfr.tar.gz

# -------------------------------------------------------
# Install ezc3d
# -------------------------------------------------------
git clone https://github.com/pyomeca/ezc3d.git
cd ezc3d
git checkout Release_1.4.7
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
make
sudo make install
cd ../..
cleanup ezc3d

# -------------------------------------------------------
# Install pybind11
# -------------------------------------------------------
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
cd ../..
rm -rf pybind11

# -------------------------------------------------------
# Install pytest and other Python packages
# -------------------------------------------------------
python3 -m pip install --break-system-packages pytest
python3 -m pip install --break-system-packages auditwheel
python3 -m pip install --break-system-packages pybind11-stubgen
python3 -m pip install --break-system-packages patchelf
python3 -m pip install --break-system-packages numpy

# -------------------------------------------------------
# Install extra tools for development
# -------------------------------------------------------
apt-get install -y gdb

# -------------------------------------------------------
# Verify protoc version
# -------------------------------------------------------
protoc --version