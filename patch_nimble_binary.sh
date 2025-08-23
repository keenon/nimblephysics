#!/bin/bash

PIP_HOME=$(pip3.11 show nimblephysics | grep "Location" | sed 's/^\(Location: \)*//')
TARGET=$PIP_HOME/nimblephysics_libs/_nimblephysics.so
rm -f $TARGET

SOURCE=build/python/_nimblephysics/_nimblephysics.so

cp $SOURCE $TARGET
python3.11 -c "import nimblephysics"