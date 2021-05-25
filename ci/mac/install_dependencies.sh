#!/bin/bash
set -e

# brew install gnu-sed
brew reinstall gcc
export FC=$(which gfortran)
echo "FC=$FC"

export MACOSX_DEPLOYMENT_TARGET="10.9"

export PYTHON3=$(which python3)
echo "Python3=${PYTHON3}"

# Install perfutils - Keenon's fork, compatible with Mac OSX
git clone https://github.com/keenon/PerfUtils.git
pushd PerfUtils
mkdir build
pushd build
cmake ..
make install
popd
popd
rm -rf PerfUtils

brew install boost # @1.73
brew install eigen

brew install openssl@1.1
cp /usr/local/opt/openssl@1.1/lib/pkgconfig/*.pc /usr/local/lib/pkgconfig/

# Install CCD
git clone https://github.com/danfis/libccd.git
pushd libccd
mkdir build
pushd build
cmake .. -DENABLE_DOUBLE_PRECISION=ON
make install -j
popd
popd
rm -rf libccd

# Install ASSIMP
git clone https://github.com/assimp/assimp.git
pushd assimp
mkdir build
pushd build
cmake ..
make install -j
popd
popd
rm -rf assimp

# Install LAPACK
brew install lapack

# Install MUMPS
git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git
pushd ThirdParty-Mumps
./get.Mumps
./configure
make -j
make install
popd
rm -rf ThirdParty-Mumps

# Install IPOPT
git clone https://github.com/coin-or/Ipopt.git
pushd Ipopt
./configure --with-mumps
make -j
make install
popd
rm -rf Ipopt
ln -s /usr/local/include/coin-or /usr/local/include/coin

# Install pybind11
git clone https://github.com/pybind/pybind11.git
pushd pybind11
mkdir build
pushd build
cmake .. -DPYTHON_EXECUTABLE:FILEPATH=$(which python3)
make install -j
popd
popd
rm -rf pybind11

# Install FCL
# Key note: this needs to happen before octomap
# git clone https://github.com/flexible-collision-library/fcl.git
# pushd fcl
# git checkout 0.3.4
# mkdir build
# pushd build
# cmake .. -DFCL_WITH_OCTOMAP=OFF -DBUILD_TESTING=OFF
# make install -j
# popd
# popd
# rm -rf fcl

# Install octomap
# git clone https://github.com/OctoMap/octomap.git
# pushd octomap
# git checkout v1.8.1
# mkdir build
# pushd build
# cmake ..
# make install -j
# popd
# popd
# rm -rf octomap

# Install tinyxml2
git clone https://github.com/leethomason/tinyxml2.git
pushd tinyxml2
mkdir build
pushd build
cmake ..
make install -j
popd
popd
rm -rf tinyxml2

# Install freeglut
# brew cask install xquartz
# brew install freeglut

# Install Open Scene Graph
# brew install open-scene-graph

# Install pytest
pip3 install pytest


# Install tinyxml1
git clone https://github.com/robotology-dependencies/tinyxml.git
pushd tinyxml
mkdir build
pushd build
cmake ..
make install -j
popd
popd
rm -rf tinyxml

# Install urdfdom_headers
git clone https://github.com/ros/urdfdom_headers.git
pushd urdfdom_headers
mkdir build
pushd build
cmake ..
make install -j
popd
popd
rm -rf urdfdom_headers

# Install console_bridge
git clone https://github.com/ros/console_bridge.git
pushd console_bridge
mkdir build
pushd build
cmake ..
make install -j
popd
popd
rm -rf console_bridge

# Install urdfdom
git clone https://github.com/ros/urdfdom.git
pushd urdfdom
mkdir build
pushd build
cmake ..
make install -j
popd
popd
rm -rf urdfdom

# Install protobuf
PROTOBUF_VERSION="3.14.0"
# wget https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz
# tar -xvzf protobuf-all-${PROTOBUF_VERSION}.tar.gz
# rm protobuf-all-${PROTOBUF_VERSION}.tar.gz
# pushd protobuf-${PROTOBUF_VERSION}
# CXX_FLAGS="-fvisibility=hidden" ./configure
# make -j
# make install
# popd
# rm -rf protobuf-${PROTOBUF_VERSION}

# Install grpc
git clone --recurse-submodules -b v1.33.2 https://github.com/grpc/grpc
pushd grpc
pushd third_party/protobuf
git checkout v${PROTOBUF_VERSION}
popd
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_CXX_FLAGS="-fvisibility=hidden" \
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
cmake -DCMAKE_BUILD_TYPE=Release ..
sudo make install
popd
popd
rm -rf benchmark

# Reset the IDs for our libraries to absolute paths
install_name_tool -id /usr/local/lib/liburdfdom_sensor.dylib /usr/local/lib/liburdfdom_sensor.dylib
install_name_tool -id /usr/local/lib/liburdfdom_model_state.dylib /usr/local/lib/liburdfdom_model_state.dylib
install_name_tool -id /usr/local/lib/liburdfdom_model.dylib /usr/local/lib/liburdfdom_model.dylib
install_name_tool -id /usr/local/lib/liburdfdom_world.dylib /usr/local/lib/liburdfdom_world.dylib
install_name_tool -id /usr/local/lib/libconsole_bridge.dylib /usr/local/lib/libconsole_bridge.dylib
install_name_tool -id /usr/local/lib/libtinyxml2.8.dylib /usr/local/lib/libtinyxml2.8.dylib
# install_name_tool -id /usr/local/lib/liboctomap.1.8.dylib /usr/local/lib/liboctomap.1.8.dylib
# install_name_tool -id /usr/local/lib/liboctomath.1.8.dylib /usr/local/lib/liboctomath.1.8.dylib
install_name_tool -id /usr/local/lib/libccd.2.dylib /usr/local/lib/libccd.2.dylib
# install_name_tool -id /usr/local/lib/libfcl.dylib /usr/local/lib/libfcl.dylib
install_name_tool -id /usr/local/lib/libassimp.5.dylib /usr/local/lib/libassimp.5.dylib
# We're not installing Open Scene Graph, so these aren't necessary
# install_name_tool -id /usr/local/lib/libosg.161.dylib /usr/local/lib/libosg.161.dylib
# install_name_tool -id /usr/local/lib/libosgViewer.161.dylib /usr/local/lib/libosgViewer.161.dylib
# install_name_tool -id /usr/local/lib/libosgManipulator.161.dylib /usr/local/lib/libosgManipulator.161.dylib
# install_name_tool -id /usr/local/lib/libosgGA.161.dylib /usr/local/lib/libosgGA.161.dylib
# install_name_tool -id /usr/local/lib/libosgDB.161.dylib /usr/local/lib/libosgDB.161.dylib
# install_name_tool -id /usr/local/lib/libosgShadow.161.dylib /usr/local/lib/libosgShadow.161.dylib
# install_name_tool -id /usr/local/lib/libOpenThreads.21.dylib /usr/local/lib/libOpenThreads.21.dylib

# Fix "icu4c" installed by Brew
ICU4C_MAJOR_VERSION="69"
ICU4C_FULL_VERSION="69.1"
pushd /usr/local/Cellar/icu4c/${ICU4C_FULL_VERSION}/lib/
sudo install_name_tool -change "@loader_path/libicuuc.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicuuc.${ICU4C_FULL_VERSION}.dylib" libicui18n.${ICU4C_FULL_VERSION}.dylib
sudo install_name_tool -change "@loader_path/libicudata.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicudata.${ICU4C_FULL_VERSION}.dylib" libicui18n.${ICU4C_FULL_VERSION}.dylib
sudo install_name_tool -change "@loader_path/libicuuc.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicuuc.${ICU4C_FULL_VERSION}.dylib" libicuio.${ICU4C_FULL_VERSION}.dylib
sudo install_name_tool -change "@loader_path/libicudata.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicudata.${ICU4C_FULL_VERSION}.dylib" libicuio.${ICU4C_FULL_VERSION}.dylib
sudo install_name_tool -change "@loader_path/libicui18n.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicui18n.${ICU4C_MAJOR_VERSION}.dylib" libicuio.${ICU4C_FULL_VERSION}.dylib
sudo install_name_tool -change "@loader_path/libicutu.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicutu.${ICU4C_FULL_VERSION}.dylib" libicutest.${ICU4C_FULL_VERSION}.dylib
sudo install_name_tool -change "@loader_path/libicui18n.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicui18n.${ICU4C_MAJOR_VERSION}.dylib" libicutest.${ICU4C_FULL_VERSION}.dylib
sudo install_name_tool -change "@loader_path/libicuuc.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicuuc.${ICU4C_FULL_VERSION}.dylib" libicutest.${ICU4C_FULL_VERSION}.dylib
sudo install_name_tool -change "@loader_path/libicudata.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicudata.${ICU4C_FULL_VERSION}.dylib" libicutest.${ICU4C_FULL_VERSION}.dylib
sudo install_name_tool -change "@loader_path/libicuuc.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicuuc.${ICU4C_FULL_VERSION}.dylib" libicutu.${ICU4C_FULL_VERSION}.dylib
sudo install_name_tool -change "@loader_path/libicudata.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicudata.${ICU4C_FULL_VERSION}.dylib" libicutu.${ICU4C_FULL_VERSION}.dylib
sudo install_name_tool -change "@loader_path/libicui18n.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicui18n.${ICU4C_MAJOR_VERSION}.dylib" libicutu.${ICU4C_FULL_VERSION}.dylib
sudo install_name_tool -change "@loader_path/libicudata.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicudata.${ICU4C_FULL_VERSION}.dylib" libicuuc.${ICU4C_FULL_VERSION}.dylib 
popd

# Get ready to bundle the links
ls /usr/local/lib/
sudo mv /usr/local/lib/libjpeg.dylib /usr/local/lib/libjpeg.old.dylib
ln -s /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libJPEG.dylib /usr/local/lib/libjpeg.lib
# sudo mv /usr/local/lib/libGIF.dylib /usr/local/lib/libGIF.old.dylib
# ln -s /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libGIF.dylib /usr/local/lib/libGIF.lib
sudo mv /usr/local/lib/libTIFF.dylib /usr/local/lib/libTIFF.old.dylib
ln -s /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libTIFF.dylib /usr/local/lib/libTIFF.lib
sudo mv /usr/local/lib/libPng.dylib /usr/local/lib/libPng.old.dylib
ln -s /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libPng.dylib /usr/local/lib/libPng.lib

# Install our build tools
python -m pip install pytest delocate