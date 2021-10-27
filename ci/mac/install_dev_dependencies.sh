#!/bin/bash

# These are dependencies we don't need on CI, but do need for local development

# Install MPFR - Arbitrary precision floating point
brew install gmp
curl https://www.mpfr.org/mpfr-current/mpfr-4.1.0.tar.gz > mpfr-4.1.0.tar.gz
tar -zxf mpfr-4.1.0.tar.gz
pushd mpfr-4.1.0
./configure
make
sudo make install
popd
rm -rf mpfr-4.1.0
rm mpfr-4.1.0.tar.gz

# Install MPIR - Arbitrary precision integer math
brew install yasm m4
brew install gcc
curl http://mpir.org/mpir-3.0.0.tar.bz2 > mpir-3.0.0.tar.bz2
tar -xf mpir-3.0.0.tar.bz2
pushd mpir-3.0.0
CC=gcc ./configure --enable-cxx
make
sudo make install
popd
rm -rf mpir-3.0.0
rm mpir-3.0.0.tar.bz2

# Install MPFRC++
wget https://github.com/advanpix/mpreal/archive/refs/tags/mpfrc++-3.6.8.tar.gz
tar -xzf mpfrc++-3.6.8.tar.gz
pushd mpreal-mpfrc-3.6.8
sudo cp mpreal.h /usr/local/include/
popd
rm -rf mpreal-mpfrc-3.6.8
rm mpfrc++-3.6.8.tar.gz