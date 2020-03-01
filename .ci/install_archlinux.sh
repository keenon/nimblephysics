#!/usr/bin/env bash
set -ex

# Sanity checks for required environment variables.
if [ -z "$BUILD_DARTPY" ]; then
  echo "Info: Environment variable BUILD_DARTPY is unset. Using OFF by default."
  BUILD_DARTPY=OFF
fi

if [ -z "$BUILD_DOCS" ]; then
  echo "Info: Environment variable BUILD_DOCS is unset. Using OFF by default."
  BUILD_DOCS=OFF
fi

if [ -z "$COMPILER" ]; then
  echo "Info: Environment variable COMPILER is unset. Using gcc by default."
  COMPILER=gcc
fi

# Build tools
yay -Syu cmake

# Required dependencies
yay -S assimp boost eigen fcl freeglut libccd libgl

# Optional dependencies
yay -S bullet coin-or-ipopt doxygen flann nlopt octomap ode openscenegraph pagmo tinyxml2 urdfdom
