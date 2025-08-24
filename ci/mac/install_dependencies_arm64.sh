#!/bin/bash
set -e

# Define the Homebrew prefix for architecture-independent paths
HOMEBREW_PREFIX=$(brew --prefix)
echo "Using Homebrew prefix: ${HOMEBREW_PREFIX}"

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

export MACOSX_DEPLOYMENT_TARGET="14.0"
export CMAKE_FLAGS="-DCMAKE_OSX_ARCHITECTURES=arm64"

export PYTHON3=$(which python3)
echo "Python3=${PYTHON3}"

# --- Homebrew Packages ---
brew install boost eigen lapack pkgconfig

# --- OpenSSL Configuration ---
brew install openssl@1.1
OPENSSL_PREFIX=$(brew --prefix openssl@1.1)
mkdir -p "${HOMEBREW_PREFIX}/lib/pkgconfig"
sudo cp "${OPENSSL_PREFIX}/lib/pkgconfig/"*.pc "${HOMEBREW_PREFIX}/lib/pkgconfig/"

# --- Build from Source ---

# Install CCD
echo "Installing libccd..."
git clone https://github.com/danfis/libccd.git
pushd libccd
git checkout v2.1
mkdir build
pushd build
cmake .. -DCMAKE_INSTALL_PREFIX="${HOMEBREW_PREFIX}" -DENABLE_DOUBLE_PRECISION=ON -DCMAKE_OSX_DEPLOYMENT_TARGET="10.15"
sudo make install -j
popd
popd
rm -rf libccd

# Install ASSIMP
echo "Installing assimp..."
git clone https://github.com/assimp/assimp.git
pushd assimp
git checkout v5.0.1
mkdir build
pushd build
cmake .. -DCMAKE_INSTALL_PREFIX="${HOMEBREW_PREFIX}"
sudo make install -j
popd
popd
rm -rf assimp

# Install MUMPS
echo "Installing MUMPS..."
git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git
pushd ThirdParty-Mumps
./get.Mumps
./configure --prefix="${HOMEBREW_PREFIX}"
sudo make install
popd
sudo rm -rf ThirdParty-Mumps

# Install IPOPT
echo "Installing Ipopt..."
git clone https://github.com/coin-or/Ipopt.git
pushd Ipopt
./configure --prefix="${HOMEBREW_PREFIX}" --with-mumps --disable-java
sudo make install -j
popd
sudo rm -rf Ipopt
# Create coin symlink in the correct prefix
sudo ln -sf "${HOMEBREW_PREFIX}/include/coin-or" "${HOMEBREW_PREFIX}/include/coin"

# Install pybind11
echo "Installing pybind11..."
git clone https://github.com/pybind/pybind11.git
pushd pybind11
git checkout v2.11.1
mkdir build
pushd build
cmake .. -DCMAKE_INSTALL_PREFIX="${HOMEBREW_PREFIX}" -DPYTHON_EXECUTABLE:FILEPATH=$(which python3.11) ${CMAKE_FLAGS}
sudo make install -j
popd
popd
sudo rm -rf pybind11

# Install tinyxml2
echo "Installing tinyxml2..."
git clone https://github.com/leethomason/tinyxml2.git
pushd tinyxml2
git checkout 8.0.0
mkdir build
pushd build
cmake .. -DCMAKE_INSTALL_PREFIX="${HOMEBREW_PREFIX}" ${CMAKE_FLAGS}
sudo make install -j
popd
popd
sudo rm -rf tinyxml2

# Install tinyxml1
echo "Installing tinyxml..."
git clone https://github.com/robotology-dependencies/tinyxml.git
pushd tinyxml
mkdir build
pushd build
cmake .. -DCMAKE_INSTALL_PREFIX="${HOMEBREW_PREFIX}" ${CMAKE_FLAGS}
sudo make install -j
popd
popd
sudo rm -rf tinyxml
file "${HOMEBREW_PREFIX}/lib/libtinyxml.2.6.2.dylib"
lipo -info "${HOMEBREW_PREFIX}/lib/libtinyxml.2.6.2.dylib"

# Install urdfdom_headers
echo "Installing urdfdom_headers..."
git clone https://github.com/ros/urdfdom_headers.git
pushd urdfdom_headers
# git checkout 1.0.5
mkdir build
pushd build
cmake .. -DCMAKE_INSTALL_PREFIX="${HOMEBREW_PREFIX}" ${CMAKE_FLAGS}
sudo make install -j
popd
popd
sudo rm -rf urdfdom_headers

# Install console_bridge
echo "Installing console_bridge..."
git clone https://github.com/ros/console_bridge.git
pushd console_bridge
git checkout 1.0.1
mkdir build
pushd build
cmake .. -DCMAKE_INSTALL_PREFIX="${HOMEBREW_PREFIX}" ${CMAKE_FLAGS}
sudo make install -j
popd
popd
sudo rm -rf console_bridge

# Install urdfdom
echo "Installing urdfdom..."
git clone https://github.com/ros/urdfdom.git
pushd urdfdom
# git checkout 3.0.0
mkdir build
pushd build
cmake .. -DCMAKE_INSTALL_PREFIX="${HOMEBREW_PREFIX}" ${CMAKE_FLAGS}
sudo make install -j
popd
popd
sudo rm -rf urdfdom
file "${HOMEBREW_PREFIX}/lib/liburdfdom_sensor.5.0.dylib"
lipo -info "${HOMEBREW_PREFIX}/lib/liburdfdom_sensor.5.0.dylib"

# Install grpc
echo "Installing grpc..."
git clone --recurse-submodules -b v1.33.2 https://github.com/grpc/grpc
pushd grpc
pushd third_party/protobuf
PROTOBUF_VERSION="3.14.0"
git checkout v${PROTOBUF_VERSION}
popd
mkdir -p cmake/build
pushd cmake/build
cmake -DCMAKE_INSTALL_PREFIX="${HOMEBREW_PREFIX}" \
      -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_OSX_DEPLOYMENT_TARGET="10.15" \
      -DCMAKE_CXX_FLAGS="-fvisibility=hidden" \
      ${CMAKE_FLAGS} \
      ../..
sudo make install -j
popd
popd
sudo rm -rf grpc

# Install Google benchmark
echo "Installing Google benchmark..."
git clone https://github.com/google/benchmark.git
git clone https://github.com/google/googletest.git benchmark/googletest
pushd benchmark
git checkout v1.8.3
pushd googletest
git checkout v1.14.0
popd
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX="${HOMEBREW_PREFIX}" -DCMAKE_BUILD_TYPE=Release ${CMAKE_FLAGS} ..
sudo make install
popd
popd
sudo rm -rf benchmark

# Install ezc3d
echo "Installing ezc3d..."
git clone https://github.com/pyomeca/ezc3d.git
pushd ezc3d
git checkout Release_1.5.4
mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX="${HOMEBREW_PREFIX}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON ${CMAKE_FLAGS} ..
sudo make install
popd
popd
sudo rm -rf ezc3d

# --- Dynamic Library Path Fixes ---

echo "Fixing dylib paths for urdfdom..."
for lib in sensor model world model_state; do
    LIB_PATH="${HOMEBREW_PREFIX}/lib/liburdfdom_${lib}.dylib"
    echo "Fixing path on ${LIB_PATH}"
    otool -L "${LIB_PATH}"
    sudo install_name_tool -change libtinyxml.2.6.2.dylib "${HOMEBREW_PREFIX}/lib/libtinyxml.2.6.2.dylib" "${LIB_PATH}"
    otool -L "${LIB_PATH}"
done

echo "Resetting dylib IDs to absolute paths..."
LIBS_TO_FIX=(
    "liburdfdom_sensor.dylib"
    "liburdfdom_model_state.dylib"
    "liburdfdom_model.dylib"
    "liburdfdom_world.dylib"
    "libconsole_bridge.dylib"
    "libtinyxml2.8.dylib"
    "libtinyxml.2.6.2.dylib"
    "libezc3d.dylib"
    "libccd.2.dylib"
    "libassimp.5.dylib"
)
for lib in "${LIBS_TO_FIX[@]}"; do
    LIB_PATH="${HOMEBREW_PREFIX}/lib/${lib}"
    if [ -f "$LIB_PATH" ]; then
        echo "Setting install_name_tool id for ${LIB_PATH}"
        sudo install_name_tool -id "${LIB_PATH}" "${LIB_PATH}"
    fi
done

echo "Fixing assimp linking issue..."
sudo install_name_tool -change "@rpath/libIrrXML.dylib" "${HOMEBREW_PREFIX}/lib/libIrrXML.dylib" "${HOMEBREW_PREFIX}/lib/libassimp.5.dylib"

echo "Fixing 'icu4c' dylib links..."
ICU4C_MAJOR_VERSION="74"
ICU4C_FULL_VERSION="74.2"
ICU4C_PREFIX=$(brew --prefix icu4c)
ICU4C_LIB_PATH="${ICU4C_PREFIX}/lib"

if [ -d "${ICU4C_LIB_PATH}" ]; then
    pushd "${ICU4C_LIB_PATH}"
    sudo install_name_tool -change "@loader_path/libicuuc.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicuuc.${ICU4C_FULL_VERSION}.dylib" "libicui18n.${ICU4C_FULL_VERSION}.dylib"
    sudo install_name_tool -change "@loader_path/libicudata.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicudata.${ICU4C_FULL_VERSION}.dylib" "libicui18n.${ICU4C_FULL_VERSION}.dylib"
    sudo install_name_tool -change "@loader_path/libicuuc.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicuuc.${ICU4C_FULL_VERSION}.dylib" "libicuio.${ICU4C_FULL_VERSION}.dylib"
    sudo install_name_tool -change "@loader_path/libicudata.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicudata.${ICU4C_FULL_VERSION}.dylib" "libicuio.${ICU4C_FULL_VERSION}.dylib"
    sudo install_name_tool -change "@loader_path/libicui18n.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicui18n.${ICU4C_MAJOR_VERSION}.dylib" "libicuio.${ICU4C_FULL_VERSION}.dylib"
    sudo install_name_tool -change "@loader_path/libicutu.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicutu.${ICU4C_FULL_VERSION}.dylib" "libicutest.${ICU4C_FULL_VERSION}.dylib"
    sudo install_name_tool -change "@loader_path/libicui18n.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicui18n.${ICU4C_MAJOR_VERSION}.dylib" "libicutest.${ICU4C_FULL_VERSION}.dylib"
    sudo install_name_tool -change "@loader_path/libicuuc.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicuuc.${ICU4C_FULL_VERSION}.dylib" "libicutest.${ICU4C_FULL_VERSION}.dylib"
    sudo install_name_tool -change "@loader_path/libicudata.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicudata.${ICU4C_FULL_VERSION}.dylib" "libicutest.${ICU4C_FULL_VERSION}.dylib"
    sudo install_name_tool -change "@loader_path/libicuuc.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicuuc.${ICU4C_FULL_VERSION}.dylib" "libicutu.${ICU4C_FULL_VERSION}.dylib"
    sudo install_name_tool -change "@loader_path/libicudata.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicudata.${ICU4C_FULL_VERSION}.dylib" "libicutu.${ICU4C_FULL_VERSION}.dylib"
    sudo install_name_tool -change "@loader_path/libicui18n.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicui18n.${ICU4C_MAJOR_VERSION}.dylib" "libicutu.${ICU4C_FULL_VERSION}.dylib"
    sudo install_name_tool -change "@loader_path/libicudata.${ICU4C_MAJOR_VERSION}.dylib" "@loader_path/libicudata.${ICU4C_FULL_VERSION}.dylib" "libicuuc.${ICU4C_FULL_VERSION}.dylib"
    sudo codesign -f -s - "libicui18n.${ICU4C_FULL_VERSION}.dylib"
    sudo codesign -f -s - "libicuio.${ICU4C_FULL_VERSION}.dylib"
    sudo codesign -f -s - "libicutest.${ICU4C_FULL_VERSION}.dylib"
    sudo codesign -f -s - "libicutu.${ICU4C_FULL_VERSION}.dylib"
    sudo codesign -f -s - "libicuuc.${ICU4C_FULL_VERSION}.dylib"
    popd
else
    echo "Warning: icu4c library path not found at ${ICU4C_LIB_PATH}. Skipping fix."
fi

echo "Linking system ImageIO libraries..."
if [ -f "/System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libJPEG.dylib" ]; then
    ls "${HOMEBREW_PREFIX}/lib/"
    sudo mv "${HOMEBREW_PREFIX}/lib/libjpeg.dylib" "${HOMEBREW_PREFIX}/lib/libjpeg.old.dylib" 2>/dev/null || true
    ln -sf /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libJPEG.dylib "${HOMEBREW_PREFIX}/lib/libjpeg.lib"
    sudo mv "${HOMEBREW_PREFIX}/lib/libTIFF.dylib" "${HOMEBREW_PREFIX}/lib/libTIFF.old.dylib" 2>/dev/null || true
    ln -sf /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libTIFF.dylib "${HOMEBREW_PREFIX}/lib/libTIFF.lib"
    sudo mv "${HOMEBREW_PREFIX}/lib/libPng.dylib" "${HOMEBREW_PREFIX}/lib/libPng.old.dylib" 2>/dev/null || true
    ln -sf /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/Resources/libPng.dylib "${HOMEBREW_PREFIX}/lib/libPng.lib"
fi

# --- Final Setup ---
echo "Installing Python build tools..."
pip3 install pytest delocate pybind11-stubgen==0.16.2 numpy torch

echo "Script finished successfully!"