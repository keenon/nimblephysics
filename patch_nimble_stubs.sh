#!/bin/bash

PIP_HOME=$(pip3 show nimblephysics | grep "Location" | sed 's/^\(Location: \)*//')
TARGET=$PIP_HOME/nimblephysics_libs

SOURCE=stubs/_nimblephysics-stubs

cp -r $SOURCE/* $TARGET/
python3 -c "import nimblephysics"