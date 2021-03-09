#!/usr/bin/env bash
set -e

if [ -f /etc/os-release ]; then
  # freedesktop.org and systemd
  . /etc/os-release
  OS=$NAME
  VER=$VERSION_ID
elif type lsb_release >/dev/null 2>&1; then
  # linuxbase.org
  OS=$(lsb_release -si)
  VER=$(lsb_release -sr)
elif [ -f /etc/lsb-release ]; then
  # For some versions of Debian/Ubuntu without lsb_release command
  . /etc/lsb-release
  OS=$DISTRIB_ID
  VER=$DISTRIB_RELEASE
elif [ -f /etc/debian_version ]; then
  # Older Debian/Ubuntu/etc.
  OS=Debian
  VER=$(cat /etc/debian_version)
elif [ -f /etc/SuSe-release ]; then
  # Older SuSE/etc.
  echo "Not supported"
  exit 1
elif [ -f /etc/redhat-release ]; then
  # Older Red Hat, CentOS, etc.
  echo "Not supported"
  exit 1
else
  # Fall back to uname, e.g. "Linux <version>", also works for BSD, etc.
  OS=$(uname -s)
  VER=$(uname -r)
fi

# Sanity checks for required environment variables.
if [ -z "$BUILD_TYPE" ]; then
  echo "Error: Environment variable BUILD_TYPE is unset."
  exit 1
fi

if [ -z "$BUILD_TEST" ]; then
  echo "Info: Environment variable BUILD_TEST is unset. Using ON by default"
  BUILD_TEST=ON
fi

if [ -z "$BUILD_BENCHMARKS" ]; then
  echo "Info: Environment variable BUILD_BENCHMARKS is unset. Using OFF by default"
  BUILD_BENCHMARKS=OFF
fi

if [ -z "$COMPILER" ]; then
  echo "Info: Environment variable COMPILER is unset. Using gcc by default."
  COMPILER=gcc
fi

if [ -z "$BUILD_DIR" ]; then
  echo "Info: Environment variable BUILD_DIR is unset. Using $PWD by default."
  BUILD_DIR=$PWD
fi

if [ -z "$CMAKE_BUILD_DIR" ]; then
  echo "Info: Environment variable CMAKE_BUILD_DIR is unset. Using .build by default."
  CMAKE_BUILD_DIR=.build
fi

# Set compilers
if [ "$COMPILER" = "gcc" ]; then
  export CC=gcc
  export CXX=g++
elif [ "$COMPILER" = "clang" ]; then
  export CC=clang
  export CXX=clang++
else
  echo "Info: Compiler isn't specified. Using the system default."
fi

# Set number of threads for parallel build
# Ref: https://unix.stackexchange.com/a/129401
if [ "$OSTYPE" = "linux-gnu" ]; then
  num_threads=$(nproc)
  if [ $num_threads -gt 3 ]; then
    num_threads="$(($num_threads - 4))"
  fi
elif [ "$OSTYPE" = "darwin" ]; then
  num_threads=$(sysctl -n hw.logicalcpu)
else
  num_threads=1
  echo "$OSTYPE is not supported to detect the number of logical CPU cores."
fi
echo "INFO: Thread count: $num_threads"

echo ""
echo "=========================================="
echo ""
echo " [ SYSTEM INFO ]"
echo ""
echo " OS       : $OS $VER ($(uname -m))"
echo " Compiler : $COMPILER $($CXX --version | perl -pe '($_)=/([0-9]+([.][0-9]+)+)/')"
echo " Cores    : $num_threads / $(nproc --all)"
echo " Memory   : $(grep MemTotal /proc/meminfo)"
echo " Build Dir: $CMAKE_BUILD_DIR"
echo " BUILD_TEST: $BUILD_TEST"
echo ""
echo "=========================================="
echo ""

# Run CMake
mkdir -p $CMAKE_BUILD_DIR && echo "" || sudo mkdir -p $CMAKE_BUILD_DIR
cd $CMAKE_BUILD_DIR
if [ "$OSTYPE" = "linux-gnu" ]; then
  install_prefix_option="-DCMAKE_INSTALL_PREFIX=/usr/"
fi
cmake --version
cmake $BUILD_DIR \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DDART_BUILD_BENCHMARKS=$BUILD_BENCHMARKS \
  -DDART_VERBOSE=ON \
  ${install_prefix_option}

# C++: build, test, and install
make -s -j$num_threads all
if [ "$BUILD_TEST" = "ON" ]; then
  make -s -j$num_threads tests
  # ctest --output-on-failure -j$num_threads
fi

# Python (_diffdart): build, test, and install
if [ "$BUILD_PYTHON" = "ON" ]; then
  make -s -j$num_threads _diffdart
fi
