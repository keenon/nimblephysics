#!/bin/bash
set -e

# Use sudo for sudo commands, but only if we're not already root
sudo ()
{
    [[ $EUID = 0 ]] || set -- command sudo "$@"
    "$@"
}

# brew install gnu-sed
brew reinstall gcc
export FC=$(which gfortran)
echo "FC=$FC"

export MACOSX_DEPLOYMENT_TARGET="15.0"
export CMAKE_FLAGS="-DCMAKE_OSX_ARCHITECTURES=x86_64"

export PYTHON3=$(which python3)
echo "Python3=${PYTHON3}"

# Install perfutils - Keenon's fork, compatible with Mac OSX
# This doesn't work with ARM on mac, since it inlines x86 instructions
# git clone https://github.com/keenon/PerfUtils.git
# pushd PerfUtils
# mkdir build
# pushd build
# cmake ..
# make install
# popd
# popd
# rm -rf PerfUtils

brew install boost # @1.73
brew install eigen

brew install openssl@1.1
if [ -d "/usr/local/opt/openssl@1.1/lib/pkgconfig/" ]; then
      # x86 Macs
      # If this fails, we skip it
      cp /usr/local/opt/openssl@1.1/lib/pkgconfig/*.pc /usr/local/lib/pkgconfig/ || :
else
      # ARM64 Macs
      # TODO: unsudo this command
      sudo mkdir -p /usr/local/lib/pkgconfig
      sudo cp /opt/homebrew/opt/openssl@1.1/lib/pkgconfig/*.pc /usr/local/lib/pkgconfig/
fi

# Install CCD
git clone https://github.com/danfis/libccd.git
pushd libccd
git checkout v2.1
mkdir build
pushd build
cmake .. -DENABLE_DOUBLE_PRECISION=ON -DCMAKE_OSX_DEPLOYMENT_TARGET="10.15"
sudo make install -j
popd
popd
rm -rf libccd

# Install ASSIMP
git clone https://github.com/assimp/assimp.git
pushd assimp
git checkout v5.0.1
mkdir build
pushd build
cmake ..
sudo make install -j
popd
popd
rm -rf assimp

# Install LAPACK
brew install lapack

# Install MUMPS
git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git
pushd ThirdParty-Mumps
./get.Mumps
./configure # CFLAGS="-arch x86_64 -arch arm64" FCFLAGS="-arch x86_64 -arch arm64" LDFLAGS="-arch x86_64 -arch arm64" 
# make # Don't build mumps in parallel, that seems to have a race-condition on the Azure CI Mac's?
sudo make install
popd
sudo rm -rf ThirdParty-Mumps

# NOTE: on local arm64 M1 mac, I've had to do the following when I get linker errors during "delocate"
# ln -s /opt/homebrew/Cellar/gcc/11.2.0_3/lib/gcc/11/libgcc_s.1.1.dylib /opt/homebrew/Cellar/gcc/11.2.0_3/lib/gcc/11/libgcc_s.1.dylib 

# Install IPOPT
git clone https://github.com/coin-or/Ipopt.git
pushd Ipopt
./configure --with-mumps --disable-java # CFLAGS="-arch x86_64 -arch arm64" FCFLAGS="-arch x86_64 -arch arm64" LDFLAGS="-arch x86_64 -arch arm64"
sudo make install -j
popd
sudo rm -rf Ipopt
sudo ln -s /usr/local/include/coin-or /usr/local/include/coin

# Install pybind11
git clone https://github.com/pybind/pybind11.git
pushd pybind11
git checkout v2.11.1
mkdir build
pushd build
cmake .. -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) $CMAKE_FLAGS
sudo make install -j
popd
popd
sudo rm -rf pybind11

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
git checkout 8.0.0
mkdir build
pushd build
cmake .. $CMAKE_FLAGS
sudo make install -j
popd
popd
sudo rm -rf tinyxml2

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
cmake .. $CMAKE_FLAGS
sudo make install -j
popd
popd
sudo rm -rf tinyxml
file /usr/local/lib/libtinyxml.2.6.2.dylib
lipo -info /usr/local/lib/libtinyxml.2.6.2.dylib

# Install urdfdom_headers
git clone https://github.com/ros/urdfdom_headers.git
pushd urdfdom_headers
mkdir build
pushd build
cmake .. $CMAKE_FLAGS
sudo make install -j
popd
popd
sudo rm -rf urdfdom_headers

# Install console_bridge
git clone https://github.com/ros/console_bridge.git
pushd console_bridge
mkdir build
pushd build
cmake .. $CMAKE_FLAGS
sudo make install -j
popd
popd
sudo rm -rf console_bridge

# Install urdfdom
git clone https://github.com/ros/urdfdom.git
pushd urdfdom
git checkout 3.0.0
mkdir build
pushd build
cmake .. $CMAKE_FLAGS
sudo make install -j
popd
popd
sudo rm -rf urdfdom
file /usr/local/lib/liburdfdom_sensor.3.0.dylib
lipo -info /usr/local/lib/liburdfdom_sensor.3.0.dylib

echo "Fixing path on liburdfdom_sensor.dylib"
otool -L /usr/local/lib/liburdfdom_sensor.dylib
sudo install_name_tool -change libtinyxml.2.6.2.dylib /usr/local/lib/libtinyxml.2.6.2.dylib /usr/local/lib/liburdfdom_sensor.dylib
otool -L /usr/local/lib/liburdfdom_sensor.dylib

echo "Fixing path on liburdfdom_model.dylib"
otool -L /usr/local/lib/liburdfdom_model.dylib
sudo install_name_tool -change libtinyxml.2.6.2.dylib /usr/local/lib/libtinyxml.2.6.2.dylib /usr/local/lib/liburdfdom_model.dylib
otool -L /usr/local/lib/liburdfdom_model.dylib

echo "Fixing path on liburdfdom_world.dylib"
otool -L /usr/local/lib/liburdfdom_world.dylib
sudo install_name_tool -change libtinyxml.2.6.2.dylib /usr/local/lib/libtinyxml.2.6.2.dylib /usr/local/lib/liburdfdom_world.dylib
otool -L /usr/local/lib/liburdfdom_world.dylib

echo "Fixing path on liburdfdom_model_state.dylib"
otool -L /usr/local/lib/liburdfdom_model_state.dylib
sudo install_name_tool -change libtinyxml.2.6.2.dylib /usr/local/lib/libtinyxml.2.6.2.dylib /usr/local/lib/liburdfdom_model_state.dylib
otool -L /usr/local/lib/liburdfdom_model_state.dylib

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

# brew install zlib
# brew install xz zlib bzip2

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
      -DCMAKE_OSX_DEPLOYMENT_TARGET="10.15" \
      -DCMAKE_CXX_FLAGS="-fvisibility=hidden" \
      $CMAKE_FLAGS \
      ../..
sudo make install -j
popd
popd
sudo rm -rf grpc

# Install Google benchmark
git clone https://github.com/google/benchmark.git
git clone https://github.com/google/googletest.git benchmark/googletest
pushd benchmark
git checkout v1.8.3
pushd googletest
git checkout v1.14.0 
popd
mkdir build
pushd build
cmake -DCMAKE_BUILD_TYPE=Release $CMAKE_FLAGS ..
sudo make install
popd
popd
sudo rm -rf benchmark

# Install ezc3d
git clone https://github.com/pyomeca/ezc3d.git
pushd ezc3d
git checkout Release_1.5.4
mkdir build
pushd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON $CMAKE_FLAGS ..
sudo make install
popd
popd
sudo rm -rf ezc3d

# Reset the IDs for our libraries to absolute paths
sudo install_name_tool -id /usr/local/lib/liburdfdom_sensor.dylib /usr/local/lib/liburdfdom_sensor.dylib
sudo install_name_tool -id /usr/local/lib/liburdfdom_model_state.dylib /usr/local/lib/liburdfdom_model_state.dylib
sudo install_name_tool -id /usr/local/lib/liburdfdom_model.dylib /usr/local/lib/liburdfdom_model.dylib
sudo install_name_tool -id /usr/local/lib/liburdfdom_world.dylib /usr/local/lib/liburdfdom_world.dylib
sudo install_name_tool -id /usr/local/lib/libconsole_bridge.dylib /usr/local/lib/libconsole_bridge.dylib
sudo install_name_tool -id /usr/local/lib/libtinyxml2.8.dylib /usr/local/lib/libtinyxml2.8.dylib
sudo install_name_tool -id /usr/local/lib/libtinyxml.2.6.2.dylib /usr/local/lib/libtinyxml.2.6.2.dylib
sudo install_name_tool -id /usr/local/lib/libezc3d.dylib /usr/local/lib/libezc3d.dylib
# install_name_tool -id /usr/local/lib/liboctomap.1.8.dylib /usr/local/lib/liboctomap.1.8.dylib
# install_name_tool -id /usr/local/lib/liboctomath.1.8.dylib /usr/local/lib/liboctomath.1.8.dylib
sudo install_name_tool -id /usr/local/lib/libccd.2.dylib /usr/local/lib/libccd.2.dylib
# install_name_tool -id /usr/local/lib/libfcl.dylib /usr/local/lib/libfcl.dylib
sudo install_name_tool -id /usr/local/lib/libassimp.5.dylib /usr/local/lib/libassimp.5.dylib
# We're not installing Open Scene Graph, so these aren't necessary
# install_name_tool -id /usr/local/lib/libosg.161.dylib /usr/local/lib/libosg.161.dylib
# install_name_tool -id /usr/local/lib/libosgViewer.161.dylib /usr/local/lib/libosgViewer.161.dylib
# install_name_tool -id /usr/local/lib/libosgManipulator.161.dylib /usr/local/lib/libosgManipulator.161.dylib
# install_name_tool -id /usr/local/lib/libosgGA.161.dylib /usr/local/lib/libosgGA.161.dylib
# install_name_tool -id /usr/local/lib/libosgDB.161.dylib /usr/local/lib/libosgDB.161.dylib
# install_name_tool -id /usr/local/lib/libosgShadow.161.dylib /usr/local/lib/libosgShadow.161.dylib
# install_name_tool -id /usr/local/lib/libOpenThreads.21.dylib /usr/local/lib/libOpenThreads.21.dylib

# An attempt to fix the assimp linking issue
sudo install_name_tool -change "@rpath/libIrrXML.dylib" "/usr/local/lib/libIrrXML.dylib" /usr/local/lib/libassimp.5.dylib 

# Different attempts to fix the liblzma linking issue
# sudo install_name_tool -id /usr/lib/liblzma.5.dylib /usr/lib/liblzma.5.dylib
# sudo install_name_tool -id /usr/lib/libcompression.dylib /usr/lib/libcompression.dylib
# brew install xz zlib bzip2

# Fix "icu4c" installed by Brew
ICU4C_MAJOR_VERSION="74"
ICU4C_FULL_VERSION="74.2"
if [ -d "/usr/local/Cellar/icu4c/${ICU4C_FULL_VERSION}/lib/" ]; then
      pushd /usr/local/Cellar/icu4c/${ICU4C_FULL_VERSION}/lib/
else
      pushd /opt/homebrew/Cellar/icu4c/${ICU4C_FULL_VERSION}/lib/
fi
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
sudo codesign -f -s - libicui18n.${ICU4C_FULL_VERSION}.dylib
sudo codesign -f -s - libicuio.${ICU4C_FULL_VERSION}.dylib
sudo codesign -f -s - libicutest.${ICU4C_FULL_VERSION}.dylib
sudo codesign -f -s - libicutu.${ICU4C_FULL_VERSION}.dylib
sudo codesign -f -s - libicuuc.${ICU4C_FULL_VERSION}.dylib
popd

# Get ready to bundle the links
if [ -f "/System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libJPEG.dylib" ]; then
    ls /usr/local/lib/
    sudo mv /usr/local/lib/libjpeg.dylib /usr/local/lib/libjpeg.old.dylib
    ln -s /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libJPEG.dylib /usr/local/lib/libjpeg.lib
    # sudo mv /usr/local/lib/libGIF.dylib /usr/local/lib/libGIF.old.dylib
    # ln -s /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libGIF.dylib /usr/local/lib/libGIF.lib
    sudo mv /usr/local/lib/libTIFF.dylib /usr/local/lib/libTIFF.old.dylib
    ln -s /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libTIFF.dylib /usr/local/lib/libTIFF.lib
    sudo mv /usr/local/lib/libPng.dylib /usr/local/lib/libPng.old.dylib
    ln -s /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libPng.dylib /usr/local/lib/libPng.lib
fi


# Replace liblzma with a hard copy of the library, instead of a link
# echo "Attempting to read LZMA links"
# readlink /usr/lib/liblzma.dylib
# readlink /usr/lib/liblzma.5.dylib
# LZMA_PATH=$(readlink /usr/lib/liblzma.dylib)
# LZMA_5_PATH=$(readlink /usr/lib/liblzma.5.dylib)
# echo "LZMA_PATH=$LZMA_PATH"
# echo "LZMA_5_PATH=$LZMA_5_PATH"
# sudo mv /usr/local/lib/liblzma.5.dylib /usr/local/lib/liblzma.5.old.dylib
# echo "Attempting to add symbolic links"
# ln -s $LZMA_PATH /usr/local/lib/liblzma.dylib
# ln -s $LZMA_5_PATH /usr/local/lib/liblzma.5.dylib
# echo "Symbolic links complete"

# Install our build tools
pip3 install pytest delocate pybind11-stubgen==0.16.2 numpy torch

# Install pkgconfig, which CMake uses to look for dependencies
brew install pkgconfig