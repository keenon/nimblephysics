#!/bin/bash
set -e

export VERSION=$(cat ../../VERSION.txt)

pushd ../..

rm -rf dist/*
rm -rf build/*
# rm -rf wheelhouse/*

# Actually build the code
python3 setup.py sdist bdist_wheel

# Install delocate, to bundle dependencies into the wheel
pushd dist
WHEEL_NAME=$(ls *.whl)
echo "WHEEL_NAME=${WHEEL_NAME}"
DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:/usr/local/lib" delocate-wheel -w ../wheelhouse -v --dylibs-only ${WHEEL_NAME}
popd

# Replace the ABI tag with a less general version of OSX, which is important for compatibility with pip on Arm64 macs
mv wheelhouse/nimblephysics-${VERSION}-cp38-cp38-macosx_10_14_arm64.whl wheelhouse/nimblephysics-${VERSION}-cp38-cp38-macosx_11_0_arm64.whl

# Install the wheel
pip3 install --force-reinstall wheelhouse/nimblephysics-${VERSION}-cp38-cp38-macosx_11_0_arm64.whl

# Actually push the wheel to PyPI
# python3 -m pip install --user --upgrade twine
# python3 -m twine upload --repository pypi wheelhouse/*