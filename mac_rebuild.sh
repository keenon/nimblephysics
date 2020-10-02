#!/bin/bash

rm -rf dist/*
rm -rf build/*
# rm -rf wheelhouse/*

# Actually build the code
python3 setup.py sdist bdist_wheel

# Install delocate, to bundle dependencies into the wheel
pushd dist
DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:/usr/local/lib" delocate-wheel -w ../wheelhouse -v diffdart-0.0.2-cp38-cp38-macosx_10_14_x86_64.whl
popd

# Replace the ABI tag with a more general version
mv wheelhouse/diffdart-0.0.2-cp38-cp38-macosx_10_14_x86_64.whl wheelhouse/diffdart-0.0.2-cp38-abi3-macosx_10_14_x86_64.whl

# install_name_tool -change "@loader_path/.dylibs/Python" "/usr/local/opt/python@3.8/Frameworks/Python.framework/Versions/3.8/Python" _diffdart.so
# install_name_tool -change "/usr/local/opt/python@3.8/Frameworks/Python.framework/Versions/3.8/Python" "Python" _diffdart.so

# Fix symlinks for ICU4
# pushd wheelhouse
# unzip
# wheel unpack diffdart-0.0.1-4-cp36-abi3-macosx_10_6_x86_64.whl
# pushd diffdart-0.0.1
# pushd diffdart_libs
# pushd .dylibs
# Add symlinks
# ln -s ./libicudata.67.1.dylib ./libicudata.67.dylib 
# ln -s ./libicuuc.67.1.dylib ./libicuuc.67.dylib 
# ln -s ./libicui18n.67.1.dylib ./libicui18n.67.dylib 
# popd
# popd
# popd
# re-zip
# mv diffdart-0.0.1-4-cp36-abi3-macosx_10_6_x86_64.whl diffdart-0.0.1-4-cp36-abi3-macosx_10_6_x86_64.tmp.whl
# wheel pack diffdart-0.0.1
# rm -rf diffdart-0.0.1
# popd

# Actually push the wheel to PyPI
# python3 -m twine upload --repository pypi wheelhouse/*
