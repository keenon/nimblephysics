#!/bin/bash
set -e

# Install Eigen
curl https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz > eigen.tar.gz
tar -zxf eigen.tar.gz
pushd eigen-3.3.7
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/bps/rl/diffdart-deps ..
make install -j14
popd
popd
rm -rf eigen-3.3.7

# Install CCD
git clone https://github.com/danfis/libccd.git
pushd libccd
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/bps/rl/diffdart-deps ..
make install -j14
popd
popd
rm -rf libccd

# Install ASSIMP
git clone https://github.com/assimp/assimp.git
pushd assimp
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/bps/rl/diffdart-deps ..
make install -j10
popd
popd
rm -rf assimp

# Install MUMPS
git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git
pushd ThirdParty-Mumps
./get.Mumps
./configure --prefix=/home/bps/rl/diffdart-deps
make -j14
make install
popd
rm -rf ThirdParty-Mumps

# Install IPOPT
git clone https://github.com/coin-or/Ipopt.git
pushd Ipopt
./configure --with-mumps --prefix=/home/bps/rl/diffdart-deps
make -j14
make install
popd
rm -rf Ipopt
ln -s /home/bps/rl/diffdart-deps/include/coin-or /home/bps/rl/diffdart-deps/include/coin

# Install pybind11
git clone https://github.com/pybind/pybind11.git
pushd pybind11
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/bps/rl/diffdart-deps ..
make install -j10
popd
popd
rm -rf pybind11

# Install FCL
# Key note: this needs to happen before octomap
git clone https://github.com/flexible-collision-library/fcl.git
pushd fcl
git checkout 0.3.4
# vi include/fcl/narrowphase/detail/convexity_based_algorithm/gjk_libccd-inl.h:1696 # "std::max(1.0, v0_dist)" -> "std::max(1.0, (double)v0_dist)"
#sed -i '1696s/v0_dist/(double)v0_dist/' include/fcl/narrowphase/detail/convexity_based_algorithm/gjk_libccd-inl.h
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/bps/rl/diffdart-deps .. -DFCL_WITH_OCTOMAP=OFF
make install -j14
popd
popd
rm -rf fcl

# Install octomap
git clone https://github.com/OctoMap/octomap.git
pushd octomap
git checkout v1.8.1
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/bps/rl/diffdart-deps ..
make install -j10
popd
popd
rm -rf octomap

# Install tinyxml2
git clone https://github.com/leethomason/tinyxml2.git
pushd tinyxml2
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/bps/rl/diffdart-deps ..
make install -j10
popd
popd
rm -rf tinyxml2

# Install Open Scene Graph
git clone https://github.com/openscenegraph/OpenSceneGraph.git
pushd OpenSceneGraph
git checkout OpenSceneGraph-3.6.5
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/bps/rl/diffdart-deps ..
make install -j10
popd
popd
rm -rf OpenSceneGraph

# Install pytest
pip install pytest

# Install tinyxml1
git clone https://github.com/robotology-dependencies/tinyxml.git
pushd tinyxml
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/bps/rl/diffdart-deps ..
make install -j10
popd
popd
rm -rf tinyxml

# Install urdfdom_headers
git clone https://github.com/ros/urdfdom_headers.git
pushd urdfdom_headers
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/bps/rl/diffdart-deps ..
make install -j10
popd
popd
rm -rf urdfdom_headers

# Install console_bridge
git clone https://github.com/ros/console_bridge.git
pushd console_bridge
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/bps/rl/diffdart-deps ..
make install -j10
popd
popd
rm -rf console_bridge

# Install urdfdom
git clone https://github.com/ros/urdfdom.git
pushd urdfdom
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/bps/rl/diffdart-deps ..
make install -j10
popd
popd
rm -rf urdfdom

# Install perfutils
git clone https://github.com/PlatformLab/PerfUtils.git
pushd PerfUtils
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/bps/rl/diffdart-deps ..
make install
popd
popd
rm -rf PerfUtils

# Install Protobuf
PROTOBUF_VERSION="3.14.0"
wget https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz
tar -xvzf protobuf-all-${PROTOBUF_VERSION}.tar.gz
rm protobuf-all-${PROTOBUF_VERSION}.tar.gz
pushd protobuf-${PROTOBUF_VERSION}
CXXFLAGS="-Wno-error=type-limits" ./configure --prefix=/home/bps/rl/diffdart-deps
make -j16
make check -j16
make install
popd
rm -rf protobuf-${PROTOBUF_VERSION}

# Install GRPC
git clone --recurse-submodules -b v1.33.2 https://github.com/grpc/grpc
pushd grpc
mkdir -p cmake/build
pushd cmake/build
cmake -DCMAKE_INSTALL_PREFIX=/home/bps/rl/diffdart-deps -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      ../..
make -j
make install
popd
popd
rm -rf grpc

# Install Google benchmark
git clone https://github.com/google/benchmark.git
git clone https://github.com/google/googletest.git benchmark/googletest
pushd benchmark
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/bps/rl/diffdart-deps -DCMAKE_BUILD_TYPE=Release ..
make install
popd
popd
rm -rf benchmark
