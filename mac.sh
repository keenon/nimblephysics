#!/bin/bash

# Install Boost from source
curl https://dl.bintray.com/boostorg/release/1.74.0/source/boost_1_74_0.tar.gz > boost.tar.gz
tar -zxf boost.tar.gz
pushd boost_1_74_0
./bootstrap.sh
./b2
./b2 install
popd
popd
rm -rf boost.tar.gz
rm -rf boost_1_74_0

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
rm -rf eigen.tar.gz

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
brew install lapack

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
# sed -i '1696s/v0_dist/(double)v0_dist/' include/fcl/narrowphase/detail/convexity_based_algorithm/gjk_libccd-inl.h
mkdir build
pushd build
cmake .. -DFCL_WITH_OCTOMAP=OFF
make install -j14
popd
popd
# rm -rf fcl

# Install octomap
git clone https://github.com/OctoMap/octomap.git
pushd octomap
git checkout v1.8.1
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
brew install freeglut
# brew cask install xquartz
# curl https://managedway.dl.sourceforge.net/project/freeglut/freeglut/3.2.1/freeglut-3.2.1.tar.gz > freeglut.tar.gz
# tar -zxf freeglut.tar.gz
# rm freeglut.tar.gz
# pushd freeglut-3.2.1
# mkdir build
# pushd build
# cmake ..
# make install -j10
# popd
# popd
# rm -rf freeglut-3.2.1

# Install Open Scene Graph
brew install open-scene-graph
# git clone https://github.com/openscenegraph/OpenSceneGraph.git
# pushd OpenSceneGraph
# git checkout OpenSceneGraph-3.6.5
# mkdir build
# pushd build
# cmake ..
# make install -j10
# popd
# popd
# rm -rf OpenSceneGraph

# Install pytest
pip3 install pytest


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

# Reset the IDs for our libraries to absolute paths
install_name_tool -id /usr/local/lib/liburdfdom_sensor.dylib /usr/local/lib/liburdfdom_sensor.dylib
install_name_tool -id /usr/local/lib/liburdfdom_model_state.dylib /usr/local/lib/liburdfdom_model_state.dylib
install_name_tool -id /usr/local/lib/liburdfdom_model.dylib /usr/local/lib/liburdfdom_model.dylib
install_name_tool -id /usr/local/lib/liburdfdom_world.dylib /usr/local/lib/liburdfdom_world.dylib
install_name_tool -id /usr/local/lib/libconsole_bridge.dylib /usr/local/lib/libconsole_bridge.dylib
install_name_tool -id /usr/local/lib/libtinyxml2.8.dylib /usr/local/lib/libtinyxml2.8.dylib
install_name_tool -id /usr/local/lib/liboctomap.1.8.dylib /usr/local/lib/liboctomap.1.8.dylib
install_name_tool -id /usr/local/lib/liboctomath.1.8.dylib /usr/local/lib/liboctomath.1.8.dylib
install_name_tool -id /usr/local/lib/libccd.2.dylib /usr/local/lib/libccd.2.dylib
install_name_tool -id /usr/local/lib/libfcl.dylib /usr/local/lib/libfcl.dylib
install_name_tool -id /usr/local/lib/libassimp.5.dylib /usr/local/lib/libassimp.5.dylib
install_name_tool -id /usr/local/lib/libosg.161.dylib /usr/local/lib/libosg.161.dylib
install_name_tool -id /usr/local/lib/libosgViewer.161.dylib /usr/local/lib/libosgViewer.161.dylib
install_name_tool -id /usr/local/lib/libosgManipulator.161.dylib /usr/local/lib/libosgManipulator.161.dylib
install_name_tool -id /usr/local/lib/libosgGA.161.dylib /usr/local/lib/libosgGA.161.dylib
install_name_tool -id /usr/local/lib/libosgDB.161.dylib /usr/local/lib/libosgDB.161.dylib
install_name_tool -id /usr/local/lib/libosgShadow.161.dylib /usr/local/lib/libosgShadow.161.dylib
install_name_tool -id /usr/local/lib/libOpenThreads.21.dylib /usr/local/lib/libOpenThreads.21.dylib

# Fix "icu4c" installed by Brew
pushd /usr/local/Cellar/icu4c/67.1/lib/
sudo install_name_tool -change "@loader_path/libicuuc.67.dylib" "@loader_path/libicuuc.67.1.dylib" libicui18n.67.1.dylib
sudo install_name_tool -change "@loader_path/libicudata.67.dylib" "@loader_path/libicudata.67.1.dylib" libicui18n.67.1.dylib
sudo install_name_tool -change "@loader_path/libicuuc.67.dylib" "@loader_path/libicuuc.67.1.dylib" libicuio.67.1.dylib
sudo install_name_tool -change "@loader_path/libicudata.67.dylib" "@loader_path/libicudata.67.1.dylib" libicuio.67.1.dylib
sudo install_name_tool -change "@loader_path/libicui18n.67.dylib" "@loader_path/libicui18n.67.dylib" libicuio.67.1.dylib
sudo install_name_tool -change "@loader_path/libicutu.67.dylib" "@loader_path/libicutu.67.1.dylib" libicutest.67.1.dylib
sudo install_name_tool -change "@loader_path/libicui18n.67.dylib" "@loader_path/libicui18n.67.dylib" libicutest.67.1.dylib
sudo install_name_tool -change "@loader_path/libicuuc.67.dylib" "@loader_path/libicuuc.67.1.dylib" libicutest.67.1.dylib
sudo install_name_tool -change "@loader_path/libicudata.67.dylib" "@loader_path/libicudata.67.1.dylib" libicutest.67.1.dylib
sudo install_name_tool -change "@loader_path/libicuuc.67.dylib" "@loader_path/libicuuc.67.1.dylib" libicutu.67.1.dylib
sudo install_name_tool -change "@loader_path/libicudata.67.dylib" "@loader_path/libicudata.67.1.dylib" libicutu.67.1.dylib
sudo install_name_tool -change "@loader_path/libicui18n.67.dylib" "@loader_path/libicui18n.67.dylib" libicutu.67.1.dylib
sudo install_name_tool -change "@loader_path/libicudata.67.dylib" "@loader_path/libicudata.67.1.dylib" libicuuc.67.1.dylib 

popd

# Install optool
 git clone git@github.com:alexzielenski/optool.git --recursive
 pushd optool
 xcodebuild
 cp build/Release/optool /usr/local/bin
 popd

# Actually build the code
python3.6 setup.py sdist bdist_wheel

# Get ready to bundle the links
sudo mv /usr/local/lib/libjpeg.dylib /usr/local/lib/libjpeg.old.dylib
ln -s /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libJPEG.dylib /usr/local/lib/libjpeg.lib
sudo mv /usr/local/lib/libGIF.dylib /usr/local/lib/libGIF.old.dylib
ln -s /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libGIF.dylib /usr/local/lib/libGIF.lib
sudo mv /usr/local/lib/libTIFF.dylib /usr/local/lib/libTIFF.old.dylib
ln -s /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libTIFF.dylib /usr/local/lib/libTIFF.lib
sudo mv /usr/local/lib/libPng.dylib /usr/local/lib/libPng.old.dylib
ln -s /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libPng.dylib /usr/local/lib/libPng.lib

# Install delocate, to bundle dependencies into the wheel
pip install delocate
pushd dist
DYLD_LIBRARY_PATH="/usr/local/lib:$DYLD_LIBRARY_PATH" delocate-wheel -w ../wheelhouse -v diffdart-0.0.1-cp36-cp36m-macosx_10_6_intel.whl
popd

# Replace the ABI tag with a more general version
mv wheelhouse/diffdart-0.0.1-cp36-cp36m-macosx_10_6_intel.whl wheelhouse/diffdart-0.0.1-4-cp36-abi3-macosx_10_6_x86_64.whl

# Install twine, to handle uploading to PyPI
python3 -m pip install --user --upgrade twine
python3 -m twine upload --repository pypi wheelhouse/*