#!/bin/sh

################################################################################
# Functions
################################################################################

# Print usage
usage()
{
    echo "usage: ./abi_check.sh <old_version_number> <new_version_number>"
    echo ""
    echo "The most commonly used commands are:"
    echo "  ./abi_check.sh v6.8.5 v6.9.0  # ABI compatibility check from v6.8.5 to v6.9.0"
    echo "  ./abi_check.sh v6.8.5         # ABI compatibility check from v6.8.5 to current code"
    echo ""
    exit 1
}

# Generate version files
gen_version_files()
{
    FILE="${BASEDIR}/abi_dart_${BRANCH_NAME}.xml"
    /bin/cat << EOM > $FILE
<version>
    ${VERSION_NUMBER}
</version>

<headers>
    ${BASEDIR}/install/dart_${BRANCH_NAME}/include/
</headers>

<libs>
    ${BASEDIR}/install/dart_${BRANCH_NAME}/lib/
</libs>
EOM
}

# Clone and build the target version of DART
build_target_version_dart()
{
    cd $BASEDIR/source
    git clone git://github.com/dartsim/dart.git -b ${BRANCH_NAME} --single-branch --depth 1 dart_${BRANCH_NAME}
    cd dart_${BRANCH_NAME}
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=${BASEDIR}/install/dart_${BRANCH_NAME} ..
    make -j6 install
}

# Build the current version of DART
build_current_dart()
{
    if [ ! -d "$BASEDIR/source/dart_current" ]; then
        mkdir $BASEDIR/source/dart_current
    fi
    cd $BASEDIR/source/dart_current
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=${BASEDIR}/install/dart_current ..
    make -j6 install
}

################################################################################
# Main routine
################################################################################

# If the number of arguments is not 1 or 2, then print usage
[ "$#" -gt 2 ] && { usage; exit 1; }
[ "$#" -lt 1 ] && { usage; exit 1; }

# Set working directory
BASEDIR="$PWD"/abi_check_work
cd ../
CURRENT_DART_DIR=$PWD
echo Current DART directory: $CURRENT_DART_DIR
echo ABI check working directory: $BASEDIR

# Create directories
if [ ! -d "$BASEDIR" ]; then
    mkdir $BASEDIR;
fi
if [ ! -d "$BASEDIR/source" ]; then
    mkdir $BASEDIR/source;
fi
if [ ! -d "$BASEDIR/install" ]; then
    mkdir $BASEDIR/install
fi

# Set variables
PRG=$0
OLD_VER=$1
if [ "$#" -eq 2 ]; then
    NEW_VER=$2
else
    NEW_VER=current
fi

# Build and install the old version of DART
BRANCH_NAME=${OLD_VER}
VERSION_NUMBER=${OLD_VER#"v"}
gen_version_files
build_target_version_dart

# Build and install the new version of DART
if [ "$#" -eq 2 ]; then
    BRANCH_NAME=${NEW_VER}
    VERSION_NUMBER=${NEW_VER#"v"}
    gen_version_files
    build_target_version_dart
else
    BRANCH_NAME=current
    VERSION_NUMBER=current
    gen_version_files
    build_current_dart
fi

# Install ABI checker
# TODO(JS): If abi-compliance-checker exists
# sudo apt-get --yes --force-yes install abi-compliance-checker

# Checkk ABI
cd $BASEDIR
abi-compliance-checker -lib DART -old abi_dart_${OLD_VER}.xml -new abi_dart_${NEW_VER}.xml
