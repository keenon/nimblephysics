#!/bin/bash

# Update the pkgconfig path
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib64/pkgconfig/

# Install CMake3
yum install -y cmake3
rm /usr/bin/cmake
ln -s /usr/bin/cmake3 /usr/bin/cmake

# Install Boost
yum install epel-release
rpm -ivh http://repo.okay.com.mx/centos/6/x86_64/release/okay-release-1-3.el6.noarch.rpm?
yum install -y boost-devel-1.55.0-25.el6.x86_64

# Install Eigen
curl https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz > eigen.tar.gz
tar -zxf eigen.tar.gz
pushd eigen-3.3.7
mkdir build
pushd build
cmake ..
make install -j14
popd
popd
rm -rf eigen-3.3.7

# Install CCD
git clone https://github.com/danfis/libccd.git
pushd libccd
mkdir build
pushd build
cmake ..
make install -j14
popd
popd
rm -rf libccd

# Install ASSIMP
git clone https://github.com/assimp/assimp.git
pushd assimp
mkdir build
pushd build
cmake ..
make install -j10
popd
popd
rm -rf assimp

# Install LAPACK
yum install -y lapack-devel

# Install MUMPS
git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git
pushd ThirdParty-Mumps
./get.Mumps
./configure
make -j14
make install
popd
rm -rf ThirdParty-Mumps

# Install IPOPT
git clone https://github.com/coin-or/Ipopt.git
pushd Ipopt
./configure --with-mumps
make -j14
make install
popd
rm -rf Ipopt
ln -s /usr/local/include/coin-or /usr/local/include/coin

# Install pybind11
git clone https://github.com/pybind/pybind11.git
pushd pybind11
mkdir build
pushd build
cmake ..
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
sed -i '1696s/v0_dist/(double)v0_dist/' include/fcl/narrowphase/detail/convexity_based_algorithm/gjk_libccd-inl.h
mkdir build
pushd build
cmake .. -DFCL_WITH_OCTOMAP=OFF
make install -j14
popd
popd
rm -rf fcl

# Install octomap
git clone https://github.com/OctoMap/octomap.git
git checkout v1.8.1
pushd octomap
mkdir build
pushd build
cmake ..
make install -j10
popd
popd
rm -rf octomap

# Install tinyxml2
git clone https://github.com/leethomason/tinyxml2.git
pushd tinyxml2
mkdir build
pushd build
cmake ..
make install -j10
popd
popd
rm -rf tinyxml2

# Install freeglut
yum install -y libXi-devel
yum install -y mesa-libGLU-devel
curl https://managedway.dl.sourceforge.net/project/freeglut/freeglut/3.2.1/freeglut-3.2.1.tar.gz > freeglut.tar.gz
tar -zxf freeglut.tar.gz
rm freeglut.tar.gz
pushd freeglut-3.2.1
mkdir build
pushd build
cmake ..
make install -j10
popd
popd
rm -rf freeglut-3.2.1

# Install Open Scene Graph
git clone https://github.com/openscenegraph/OpenSceneGraph.git
pushd OpenSceneGraph
git checkout OpenSceneGraph-3.6.5
mkdir build
pushd build
cmake ..
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
cmake ..
make install -j10
popd
popd
rm -rf tinyxml

# Install urdfdom_headers
git clone https://github.com/ros/urdfdom_headers.git
pushd urdfdom_headers
mkdir build
pushd build
cmake ..
make install -j10
popd
popd
rm -rf urdfdom_headers

# Install console_bridge
git clone https://github.com/ros/console_bridge.git
pushd console_bridge
mkdir build
pushd build
cmake ..
make install -j10
popd
popd
rm -rf console_bridge

# Install urdfdom
git clone https://github.com/ros/urdfdom.git
pushd urdfdom
mkdir build
pushd build
cmake ..
make install -j10
popd
popd
rm -rf urdfdom

# Install perfutils
git clone https://github.com/PlatformLab/PerfUtils.git
pushd PerfUtils
mkdir build
pushd build
cmake ..
make install
popd
popd
rm -rf PerfUtils

/opt/python/cp38-cp38/bin/python3.8
# Actually build the code
python3 setup.py sdist bdist_wheel
auditwheel repair dist/diffdart-0.0.1-cp36-cp36m-linux_x86_64.whl
# Ensure we have the ABI3 tag, so that pip interpreters will accept us
mv wheelhouse/diffdart-0.0.1-cp36-cp36m-manylinux2010_x86_64.whl wheelhouse/diffdart-0.0.1-cp36-abi3-manylinux2010_x86_64.whl